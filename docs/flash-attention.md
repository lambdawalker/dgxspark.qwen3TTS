# FlashAttention on the DGX Spark GB10

The official `flash-attn` distribution has no wheel for
`linux_aarch64 + sm_121`, and FlashAttention 2 does not list SM 12.1 as a target.
This repository can build a local FA2 wheel for the Spark by adding
`compute_121, code=sm_121` to its CUDA architecture list.

## Build

```bash
cd /home/spark/opt/TTS/dgxspark.qwen3TTS
./scripts/build_flash_attn.sh
```

The script:

1. downloads `flash-attn==2.8.3.post2` from PyPI and verifies its SHA-256,
2. applies [`scripts/patches/flash-attn-2.8.3.post2-sm121.patch`](../scripts/patches/flash-attn-2.8.3.post2-sm121.patch),
3. builds only the SM 12.1 image with `FLASH_ATTN_CUDA_ARCHS=121`,
4. runs at `nice -n 19` with `MAX_JOBS=4` and `NVCC_THREADS=2`, and
5. writes a wheel to `wheels/`.

The defaults are deliberately conservative. A high-parallelism CUDA build can
starve the 20 ARM cores and make the Spark unreachable. Override only with a
reason:

```bash
MAX_JOBS=2 ./scripts/build_flash_attn.sh
```

## Install

```bash
uv pip install --python .venv/bin/python --no-deps wheels/flash_attn-*.whl
.venv/bin/python -c 'import flash_attn; print(flash_attn.__version__)'
```

`start.sh` installs the local wheel automatically when
`QWEN_TTS_ATTENTION=flash_attention_2` and the wheel is present.

Do not run a plain `uv sync` after installing the wheel: the wheel is not in
`uv.lock`, so uv removes it. Use `./start.sh`, or reinstall the wheel after a
sync. The launcher synchronizes locked dependencies on every start using
`uv sync --frozen --extra api --inexact`, which preserves the extra wheel.
The default remains SDPA, which is in the lock and needs no local build.

## Select it

```bash
QWEN_TTS_ATTENTION=flash_attention_2 ./start.sh
```

The API advertises the selected runtime in `/v1/models` as
`native-12hz/cuda:0/bfloat16/flash_attention_2`, and job provenance records the
same value.

## Verify the kernel

Before exercising the full model:

```bash
.venv/bin/python - <<'PY'
import torch
from flash_attn import flash_attn_func

q = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
out = flash_attn_func(q, q, q, causal=True)
torch.cuda.synchronize()
print(out.shape, out.dtype, torch.isfinite(out).all().item())
PY
```

A successful import does not prove the kernel image is usable. The wheel must
contain an SM 12.1 image; `cuobjdump --list-elf` on the installed
`flash_attn_2_cuda` extension should list `sm_121`.

## What to expect

FlashAttention on SM 12.1 is a compatibility build, not an official optimized
Blackwell target. Reports from the FlashAttention issue tracker found FA2
kernels compiling and running on GB10 but tying or losing to PyTorch SDPA at
short sequence lengths. This repository measures the actual TTS workload rather
than assuming a gain; see the DGX Spark performance report for the result.

References:

- [FlashAttention issue #1969: SM 12.1 / GB10 support](https://github.com/Dao-AILab/flash-attention/issues/1969)
- [DGX Spark performance notes](https://haoailab.com/FastVideo/getting_started/installation/spark_performance)
