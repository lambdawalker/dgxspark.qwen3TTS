"""Verify pinning and native defaults without importing CUDA dependencies."""

import ast
import sys
import types
from pathlib import Path

from qwen_tts_api.profiles import SPECS
from qwen_tts_api.runtime import load_model


def test_loader_uses_one_snapshot_for_all_components(monkeypatch):
    calls = []
    sentinel = object()
    torch = types.ModuleType("torch")
    torch.bfloat16 = sentinel
    hub = types.ModuleType("huggingface_hub")

    def snapshot(repo, **kwargs):
        calls.append(("snapshot", repo, kwargs))
        return "/cache/snapshots/" + "a" * 40

    hub.snapshot_download = snapshot
    native = types.ModuleType("qwen_tts")

    class Model:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append(("load", path, kwargs))
            return sentinel

    native.Qwen3TTSModel = Model
    for name, module in [
        ("torch", torch),
        ("huggingface_hub", hub),
        ("qwen_tts", native),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    assert load_model(SPECS["qwen3-tts-1.7b-base"], "a" * 40) is sentinel
    assert calls == [
        ("snapshot", "Qwen/Qwen3-TTS-12Hz-1.7B-Base", {"revision": "a" * 40}),
        (
            "load",
            "/cache/snapshots/" + "a" * 40,
            {
                "device_map": "cuda:0",
                "dtype": sentinel,
                "attn_implementation": "sdpa",
                "local_files_only": True,
            },
        ),
    ]


def test_native_generation_default_merging_matches_adapter_call():
    # Execute the actual pure-Python merging method, without importing the model's
    # Torch decorators and heavy module imports. This detects upstream API drift.
    path = Path(__file__).resolve().parents[2] / "qwen_tts/inference/qwen3_tts_model.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen3TTSModel")
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_merge_generate_kwargs"
    )
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {}
    code = ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[]))
    exec(compile(code, str(path), "exec"), namespace)
    resolved = namespace["_merge_generate_kwargs"](
        types.SimpleNamespace(generate_defaults={"temperature": 0.7}), top_k=12
    )
    assert resolved["top_k"] == 12 and resolved["temperature"] == 0.7
    assert resolved["max_new_tokens"] == 2048 and resolved["subtalker_dosample"] is True
