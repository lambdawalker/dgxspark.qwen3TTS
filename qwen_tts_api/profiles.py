"""Capability declarations for the five official Qwen3-TTS 12Hz checkpoints."""

from dataclasses import dataclass

from tts_api_server.adapter import Profile
from tts_api_server.models import Voice

from . import __version__

LANGUAGES = {
    "auto": "Auto",
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}
SPEAKERS = (
    "vivian",
    "serena",
    "uncle_fu",
    "dylan",
    "eric",
    "ryan",
    "aiden",
    "ono_anna",
    "sohee",
)


@dataclass(frozen=True)
class ModelSpec:
    id: str
    checkpoint: str
    kind: str
    native_size: str


SPECS = {}
for size, family, kind in (
    ("0.6B", "Base", "base"),
    ("1.7B", "Base", "base"),
    ("0.6B", "CustomVoice", "custom_voice"),
    ("1.7B", "CustomVoice", "custom_voice"),
    ("1.7B", "VoiceDesign", "voice_design"),
):
    identifier = f"qwen3-tts-{size.lower()}-{family.lower()}"
    SPECS[identifier] = ModelSpec(
        identifier,
        f"Qwen/Qwen3-TTS-12Hz-{size}-{family}",
        kind,
        "0b6" if size == "0.6B" else "1b7",
    )


def extension_schema(spec):
    properties = {
        "do_sample": {"type": "boolean"},
        "top_k": {"type": "integer", "minimum": 0},
        "top_p": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
        "temperature": {"type": "number", "exclusiveMinimum": 0},
        "repetition_penalty": {"type": "number", "exclusiveMinimum": 0},
        "subtalker_dosample": {"type": "boolean"},
        "subtalker_top_k": {"type": "integer", "minimum": 0},
        "subtalker_top_p": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
        "subtalker_temperature": {"type": "number", "exclusiveMinimum": 0},
        "max_new_tokens": {"type": "integer", "minimum": 1, "maximum": 8192},
        "non_streaming_mode": {"type": "boolean"},
    }
    if spec.kind == "base":
        properties["x_vector_only_mode"] = {"type": "boolean"}
    return {"type": "object", "additionalProperties": False, "properties": properties}


def make_profile(spec, revision):
    base = spec.kind == "base"
    custom = spec.kind == "custom_voice"
    design = spec.kind == "voice_design"
    return Profile(
        id=spec.id,
        description=f"{spec.checkpoint}: {spec.kind}",
        checkpoint=spec.checkpoint,
        checkpoint_revision=revision,
        adapter_version=__version__,
        inference_mode="native-12hz",
        features={
            "tts": not design,
            "voice_cloning": base,
            "voice_design": design,
            "voice_conversion": False,
        },
        instructions=custom and spec.native_size == "1b7",
        languages=list(LANGUAGES),
        voice_required=not design,
        voices=[
            Voice(
                id=f"{spec.id}:{name}",
                alias=name,
                model=spec.id,
                kind="preset",
                provenance={"native_speaker": name},
            )
            for name in SPEAKERS
        ]
        if custom
        else [],
        output_formats=["wav", "flac"],
        sample_rates=[24000],
        reference_transcript_required=False,
        max_reference_count=1,
        max_reference_duration=60,
        feature_combinations=[
            {
                "features": ["tts", "voice_cloning"],
                "instructions": False,
                "transcript_required_unless": "extensions.qwen.x_vector_only_mode=true",
            }
        ]
        if base
        else [],
        extension_schemas={"qwen": extension_schema(spec)},
        native_incremental_audio=False,
        cancellation="cooperative_after_inference",
        progress="stages",
    )
