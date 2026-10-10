import io
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from tts_api_server.adapter import ExecutionContext
from tts_api_server.errors import DomainError
from tts_api_server.models import Reference, Voice

from qwen_tts_api.adapter import QwenAdapter
from qwen_tts_api.profiles import SPEAKERS, SPECS


class NativeModel:
    def __init__(self, spec):
        self.model = SimpleNamespace(tts_model_type=spec.kind, tts_model_size=spec.native_size)
        self.calls = []

    def get_supported_languages(self):
        return [
            "auto",
            "chinese",
            "english",
            "japanese",
            "korean",
            "german",
            "french",
            "russian",
            "portuguese",
            "spanish",
            "italian",
        ]

    def get_supported_speakers(self):
        return list(SPEAKERS)

    def _merge_generate_kwargs(self, **kwargs):
        return {"max_new_tokens": 2048, "temperature": 0.9, **kwargs}

    def _generate(self, method, **kwargs):
        self.calls.append((method, kwargs))
        return [np.zeros(2400, dtype=np.float32)], 24000

    def generate_custom_voice(self, **kwargs):
        return self._generate("custom_voice", **kwargs)

    def generate_voice_design(self, **kwargs):
        return self._generate("voice_design", **kwargs)

    def generate_voice_clone(self, **kwargs):
        return self._generate("voice_clone", **kwargs)


@pytest.fixture
def adapter():
    return QwenAdapter(
        {key: "a" * 40 for key in SPECS},
        loader=lambda spec, revision: NativeModel(spec),
        releaser=lambda: None,
    )


def test_discovery_is_lazy_and_size_specific(adapter):
    profiles = {p.id: p for p in adapter.profiles()}
    assert adapter.loaded_model is None
    assert len(profiles) == 5
    assert profiles["qwen3-tts-1.7b-customvoice"].instructions
    assert not profiles["qwen3-tts-0.6b-customvoice"].instructions
    assert not profiles["qwen3-tts-1.7b-base"].instructions
    assert not profiles["qwen3-tts-1.7b-voicedesign"].features["tts"]
    assert all(not p.features["voice_conversion"] for p in profiles.values())


def test_custom_voice_preserves_content_and_native_options(adapter):
    p = "qwen3-tts-1.7b-customvoice"
    voice = next(v for v in adapter.profiles()[3].voices if v.alias == "ryan")
    request = {
        "text": "  [laugh] Héllo!\n",
        "instructions": "  Gently.\n",
        "language": "en",
        "output": {"format": "wav"},
        "extensions": {"qwen": {"top_k": 12}},
    }
    adapter.load(p)
    outputs = adapter.execute(
        "tts",
        p,
        request,
        voice,
        ExecutionContext(lambda *a: None, lambda: False, lambda x: None),
    )
    method, args = adapter.loaded_model.calls[-1]
    assert method == "custom_voice"
    assert args["text"] == request["text"]
    assert args["instruct"] == request["instructions"]
    assert args["speaker"] == "ryan" and args["language"] == "English"
    assert args["top_k"] == 12
    assert sf.info(io.BytesIO(outputs[0].data)).samplerate == 24000
    assert outputs[0].provenance["native_parameters"]["top_k"] == 12


def test_clone_transcript_mode_and_audio_conversion(adapter, tmp_path):
    p = "qwen3-tts-1.7b-base"
    voice = Voice(
        id="v",
        model=p,
        references=[Reference(asset_id="a", transcript="  Exact words.\n")],
    )
    path = tmp_path / "ref.wav"
    sf.write(path, np.zeros((1600, 2)), 16000)
    request = {"text": "Hello", "output": {"format": "flac"}, "extensions": {}}
    adapter.validate("tts", p, request, voice)
    adapter.load(p)
    out = adapter.execute(
        "tts",
        p,
        request,
        voice,
        ExecutionContext(lambda *a: None, lambda: False, lambda x: path),
    )
    method, args = adapter.loaded_model.calls[-1]
    assert method == "voice_clone" and args["ref_text"] == "  Exact words.\n"
    assert args["ref_audio"][0].ndim == 1 and args["ref_audio"][1] == 16000
    assert sf.info(io.BytesIO(out[0].data)).format == "FLAC"
    voice.references[0].transcript = None
    with pytest.raises(DomainError, match="transcript"):
        adapter.validate("tts", p, request, voice)
    request["extensions"] = {"qwen": {"x_vector_only_mode": True}}
    adapter.validate("tts", p, request, voice)
    adapter.validate("voice_registration", p, {}, voice)


def test_design_and_cancellation(adapter):
    p = "qwen3-tts-1.7b-voicedesign"
    adapter.load(p)
    request = {
        "description": "  Warm voice. ",
        "preview_text": "  Sample.\n",
        "output": {"format": "wav"},
    }
    context = ExecutionContext(lambda *a: None, lambda: False, lambda x: None)
    adapter.execute("voice_design", p, request, None, context)
    _, args = adapter.loaded_model.calls[-1]
    assert args["instruct"] == request["description"] and args["text"] == request["preview_text"]
    assert (
        adapter.execute(
            "voice_design",
            p,
            request,
            None,
            ExecutionContext(lambda *a: None, lambda: True, lambda x: None),
        )
        == []
    )
    assert len(adapter.loaded_model.calls) == 1


def test_load_rejects_checkpoint_family_mismatch():
    adapter = QwenAdapter(
        {"qwen3-tts-1.7b-base": "a" * 40},
        loader=lambda *a: NativeModel(SPECS["qwen3-tts-0.6b-customvoice"]),
        releaser=lambda: None,
    )
    with pytest.raises(DomainError, match="checkpoint"):
        adapter.load("qwen3-tts-1.7b-base")
    assert adapter.loaded_model is None
