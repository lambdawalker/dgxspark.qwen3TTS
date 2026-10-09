"""Qwen native inference behind the shared HTTP/MCP application services."""

import io
import re

import numpy as np
import soundfile as sf
from tts_api_server.adapter import Adapter, AudioOutput
from tts_api_server.errors import DomainError

from .profiles import LANGUAGES, SPEAKERS, SPECS, make_profile
from .runtime import load_model, release_model


class QwenAdapter(Adapter):
    def __init__(
        self, revisions, *, loader=load_model, releaser=release_model, runtime_options=None
    ):
        if not revisions or any(key not in SPECS for key in revisions):
            raise ValueError("Select at least one known Qwen profile")
        if any(not re.fullmatch(r"[0-9a-f]{40}", value) for value in revisions.values()):
            raise ValueError("Checkpoint revisions must be resolved 40-character commit hashes")
        self.revisions = dict(revisions)
        self._profiles = {key: make_profile(SPECS[key], rev) for key, rev in revisions.items()}
        self.runtime_options = dict(runtime_options or {})
        if self.runtime_options:
            mode = "/".join(str(self.runtime_options[k]) for k in ("device", "dtype", "attention"))
            for profile in self._profiles.values():
                profile.inference_mode = f"native-12hz/{mode}"
        self._loader, self._releaser = loader, releaser
        self.loaded_model = None
        self.loaded_profile = None

    def profiles(self):
        return [p.model_copy(deep=True) for p in self._profiles.values()]

    def guidance(self, profile, feature):
        spec = SPECS[profile]
        example = {"model": profile, "text": "Hello from Qwen.", "language": "en"}
        if spec.kind == "base":
            example["voice"] = {"id": "<registered-voice-id>"}
        elif spec.kind == "custom_voice":
            example["voice"] = {"alias": "ryan"}
        else:
            example = {
                "model": profile,
                "description": "A warm, calm narrator.",
                "preview_text": "Hello from Qwen.",
                "language": "en",
            }
        if feature == "voice_cloning":
            example = {
                "model": profile,
                "references": [
                    {
                        "asset_id": "<uploaded-asset-id>",
                        "transcript": "Exact words in the recording.",
                    }
                ],
            }
        return {
            "summary": f"Native {spec.checkpoint} {feature}.",
            "input_rules": [
                "Text, instructions, descriptions and transcripts pass unchanged to Qwen.",
                "Use language codes from capabilities; omitted language means native Auto.",
                (
                    "Base requires one registered reference. ICL needs a nonempty exact transcript; "
                    "x_vector_only_mode=true allows no transcript, with reduced cloning quality."
                ),
                "CustomVoice requires an explicit preset voice. Only 1.7B supports instructions.",
                "Register a VoiceDesign candidate explicitly against a Base profile to reuse it.",
                "extensions.qwen accepts only the published native controls.",
            ],
            "examples": [example],
            "limitations": [
                "No voice conversion, seed control, application tag interpretation or incremental audio.",
                "Lifecycle stages are available; token percentages are not measured.",
                "Cancellation takes effect at native-call boundaries; it does not interrupt the GPU call.",
                "Reference channels are averaged to mono; native speaker/tokenizer encoders may resample.",
                "Qwen performs its own chat-template formatting and tokenization.",
                "One model is loaded at a time. No GPU/acoustic conformance is claimed by CPU tests.",
            ],
            "sources": [
                f"https://huggingface.co/{spec.checkpoint}/tree/{self.revisions[profile]}",
                "https://github.com/lambdawalker/dgxspark.qwen3TTS/blob/main/docs/api.md",
            ],
        }

    def validate(self, operation, profile, request, voice):
        spec = SPECS[profile]
        if operation == "voice_registration":
            if spec.kind != "base" or voice is None or len(voice.references) != 1:
                raise DomainError("invalid_reference", "Base requires exactly one reference.")
            return
        if operation not in {"tts", "voice_design"} or (operation == "voice_design") != (
            spec.kind == "voice_design"
        ):
            raise DomainError(
                "unsupported_feature", "This checkpoint does not support the operation."
            )
        if request.get("instructions") is not None and not self._profiles[profile].instructions:
            raise DomainError(
                "unsupported_parameter",
                "This checkpoint does not support instructions.",
                "instructions",
            )
        if spec.kind == "base":
            if voice is None or voice.kind != "reference" or len(voice.references) != 1:
                raise DomainError(
                    "invalid_reference",
                    "Base requires exactly one registered reference.",
                )
            xvector = request.get("extensions", {}).get("qwen", {}).get("x_vector_only_mode", False)
            if not xvector and not voice.references[0].transcript:
                raise DomainError(
                    "invalid_reference",
                    "ICL cloning requires the reference transcript.",
                    "voice",
                )
        if spec.kind == "custom_voice" and (
            voice is None
            or voice.kind != "preset"
            or voice.id not in {v.id for v in self._profiles[profile].voices}
        ):
            raise DomainError(
                "voice_unavailable",
                "Select a preset voice for this checkpoint.",
                "voice",
            )

    def load(self, profile):
        if self.loaded_profile == profile:
            return
        if self.loaded_profile is not None:
            self.unload(self.loaded_profile)
        spec = SPECS[profile]
        model = None
        try:
            model = self._loader(spec, self.revisions[profile])
            if (
                model.model.tts_model_type != spec.kind
                or model.model.tts_model_size != spec.native_size
                or not {v.lower() for v in LANGUAGES.values()}.issubset(
                    set(model.get_supported_languages() or [])
                )
                or (
                    spec.kind == "custom_voice"
                    and not set(SPEAKERS).issubset(set(model.get_supported_speakers() or []))
                )
            ):
                raise DomainError(
                    "model_unavailable",
                    "Loaded checkpoint does not match the advertised profile.",
                )
        except Exception as exc:
            model = None
            self._releaser()
            if isinstance(exc, DomainError):
                raise
            raise DomainError(
                "model_unavailable",
                "Unable to load the pinned Qwen checkpoint.",
                retryable=True,
            ) from exc
        self.loaded_model, self.loaded_profile = model, profile

    def unload(self, profile):
        self.loaded_model = None
        self.loaded_profile = None
        self._releaser()

    def execute(self, operation, profile, request, voice, context):
        self.validate(operation, profile, request, voice)
        if context.cancelled():
            return []
        if self.loaded_profile != profile:
            raise DomainError("model_unavailable", "The requested model is not loaded.")
        spec, model = SPECS[profile], self.loaded_model
        options = dict(request.get("extensions", {}).get("qwen", {}))
        xvector = options.pop("x_vector_only_mode", False)
        non_streaming = options.pop("non_streaming_mode", spec.kind != "base")
        parameters = model._merge_generate_kwargs(**options)
        kwargs = dict(
            parameters,
            non_streaming_mode=non_streaming,
            language=LANGUAGES[request.get("language") or "auto"],
        )
        preparation = []
        if spec.kind == "base":
            context.emit("preparing_reference", None)
            ref = voice.references[0]
            samples, rate = sf.read(
                context.asset_path(ref.asset_id), dtype="float32", always_2d=True
            )
            channels = samples.shape[1]
            samples = samples.mean(axis=1)
            preparation.append(
                {
                    "operation": "decode_reference",
                    "source_sample_rate": rate,
                    "source_channels": channels,
                    "channel_mapping": "mean_to_mono",
                    "native_resampling": "speaker encoder and speech tokenizer requirements",
                }
            )
            kwargs.update(
                text=request["text"],
                ref_audio=(samples, rate),
                ref_text=ref.transcript,
                x_vector_only_mode=xvector,
            )
            generate = model.generate_voice_clone
        elif spec.kind == "custom_voice":
            kwargs.update(text=request["text"], speaker=voice.id.split(":", 1)[1])
            if "instructions" in request:
                kwargs["instruct"] = request["instructions"]
            generate = model.generate_custom_voice
        else:
            kwargs.update(text=request["preview_text"], instruct=request["description"])
            generate = model.generate_voice_design
        if context.cancelled():
            return []
        context.emit("native_inference", None)
        waves, rate = generate(**kwargs)
        if context.cancelled():
            return []
        if rate != 24000 or len(waves) != 1:
            raise DomainError("inference_failed", "Unexpected native audio result.")
        wave = np.asarray(waves[0])
        if wave.ndim != 1 or not wave.size or not np.isfinite(wave).all():
            raise DomainError("inference_failed", "The model returned invalid audio samples.")
        context.emit("encoding", None)
        format_name = request.get("output", {}).get("format", "wav")
        buffer = io.BytesIO()
        sf.write(buffer, wave, rate, format=format_name.upper(), subtype="PCM_16")
        return [
            AudioOutput(
                buffer.getvalue(),
                format_name,
                {
                    "native_method": generate.__name__,
                    "native_parameters": dict(
                        parameters,
                        non_streaming_mode=non_streaming,
                        **({"x_vector_only_mode": xvector} if spec.kind == "base" else {}),
                    ),
                    "native_language": kwargs["language"],
                    "native_speaker": kwargs.get("speaker"),
                    "runtime": self.runtime_options,
                    "audio_preparation": preparation,
                    "output_encoding": f"{format_name}/PCM_16",
                },
            )
        ]

    def translate_error(self, error):
        if isinstance(error, DomainError):
            return error
        # Do not return native paths, caller text or credentials in public errors.
        return DomainError(
            "inference_failed", "Qwen could not generate audio. Check the server log."
        )
