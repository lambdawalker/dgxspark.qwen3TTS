"""Heavy dependencies are imported only when a worker loads/releases a model."""

import gc


def load_model(spec, revision, *, device="cuda:0", dtype="bfloat16", attention="sdpa"):
    import torch
    from huggingface_hub import snapshot_download

    from qwen_tts import Qwen3TTSModel

    # Native from_pretrained does not forward revision to every component.
    # A single immutable local snapshot keeps weights, processor and tokenizer aligned.
    snapshot = snapshot_download(spec.checkpoint, revision=revision)
    return Qwen3TTSModel.from_pretrained(
        snapshot,
        device_map=device,
        dtype=getattr(torch, dtype),
        attn_implementation=attention,
        local_files_only=True,
    )


def release_model():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
