"""Deployment configuration for the shared server's zero-argument adapter factory."""

import json
import os
import re
from functools import partial

from .adapter import QwenAdapter
from .profiles import SPECS
from .runtime import load_model

DEFAULT_PROFILES = (
    "qwen3-tts-1.7b-customvoice",
    "qwen3-tts-1.7b-base",
    "qwen3-tts-1.7b-voicedesign",
    "qwen3-tts-0.6b-customvoice",
    "qwen3-tts-0.6b-base",
)

# Official Hub commits inspected when the integration was added. Updating these
# is an intentional capability/conformance change, not an implicit main update.
DEFAULT_REVISIONS = {
    "qwen3-tts-0.6b-base": "5d83992436eae1d760afd27aff78a71d676296fc",
    "qwen3-tts-1.7b-base": "fd4b254389122332181a7c3db7f27e918eec64e3",
    "qwen3-tts-0.6b-customvoice": "85e237c12c027371202489a0ec509ded67b5e4b5",
    "qwen3-tts-1.7b-customvoice": "0c0e3051f131929182e2c023b9537f8b1c68adfe",
    "qwen3-tts-1.7b-voicedesign": "5ecdb67327fd37bb2e042aab12ff7391903235d3",
}


def resolve_revision(checkpoint, revision):
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    from huggingface_hub import HfApi

    try:
        return HfApi().model_info(checkpoint, revision=revision, timeout=30).sha
    except Exception as exc:
        raise ValueError(
            f"Cannot resolve checkpoint revision for {checkpoint}. Check Hub access, or set "
            "QWEN_TTS_REVISIONS to cached commit hashes for offline operation."
        ) from exc


def create_adapter(environment=None, *, resolver=resolve_revision):
    environment = os.environ if environment is None else environment
    selected = environment.get("QWEN_TTS_PROFILES", ",".join(DEFAULT_PROFILES)).split(",")
    selected = [item.strip() for item in selected]
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(key not in SPECS for key in selected)
    ):
        raise ValueError(
            "QWEN_TTS_PROFILES must be a comma-separated list of unique known profile IDs"
        )
    revisions = json.loads(environment.get("QWEN_TTS_REVISIONS", "{}"))
    if not isinstance(revisions, dict) or any(
        key not in SPECS or not isinstance(value, str) or not value
        for key, value in revisions.items()
    ):
        raise ValueError("QWEN_TTS_REVISIONS must map known profile IDs to Hub revisions")
    dtype = environment.get("QWEN_TTS_DTYPE", "bfloat16")
    attention = environment.get("QWEN_TTS_ATTENTION", "sdpa")
    if dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError("QWEN_TTS_DTYPE must be bfloat16, float16 or float32")
    if attention not in {"sdpa", "eager", "flash_attention_2"}:
        raise ValueError("QWEN_TTS_ATTENTION must be sdpa, eager or flash_attention_2")
    pinned = {
        key: resolver(SPECS[key].checkpoint, revisions.get(key, DEFAULT_REVISIONS[key]))
        for key in selected
    }
    runtime_options = {
        "device": environment.get("QWEN_TTS_DEVICE", "cuda:0"),
        "dtype": dtype,
        "attention": attention,
    }
    loader = partial(load_model, **runtime_options)
    return QwenAdapter(pinned, loader=loader, runtime_options=runtime_options)
