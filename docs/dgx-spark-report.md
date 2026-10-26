# Qwen3-TTS DGX Spark performance report

**Date:** 2026-10-26  
**Hardware:** NVIDIA DGX Spark, GB10, SM 12.1, 121 GiB unified memory  
**Repository:** `dgxspark.qwen3TTS`  
**Scope:** native Qwen3-TTS 1.7B through the shared HTTP/MCP server, with a
FlashAttention 2 compatibility build for SM 12.1.

## Executive summary

The existing fork already had the right deployment shape: Python 3.13, Torch
2.10/CUDA 13 ARM64, pinned model revisions, a shared HTTP/MCP server, and SDPA
inference. The missing optimization was FlashAttention, which has no official
`linux_aarch64 + sm_121` wheel.

The native 1.7B CustomVoice path is already faster than real time on GB10:
**RTF 0.741–0.798** for 2–12 s clips once the model is resident. Cold model load
adds about **37 s**, and the resident model uses about **4.5 GiB** of unified
memory.

A local FA2 2.8.3.post2 wheel can be built with an explicit `compute_121/sm_121`
target. The build script and patch are included. The measured comparison against
SDPA is recorded below; SDPA remains the default unless a measured run shows a
real gain.

## Hardware and software

| Component | Value |
| --- | --- |
| GPU | NVIDIA GB10 |
| Compute capability | 12.1 (SM 121) |
| Unified memory | 121 GiB |
| Driver | 580.173.02 |
| CUDA runtime | 13.0 |
| Python | 3.13.15 |
| PyTorch | 2.10.0+cu130 |
| Transformers | 4.57.3 |
| Model | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| Runtime | `cuda:0`, BF16, SDPA, 24 kHz mono |

The model checkpoint is pinned at
`0c0e3051f131929182e2c023b9537f8b1c68adfe` and the tokenizer checkpoint is
`7dd38ad4e9bad454aae9cd937d0cd577604fe229`.

## What changed in this fork

The shared server, adapter and API contract were already implemented and were
not duplicated. This pass adds:

1. `start.sh` — one-command HTTP + MCP launch with anonymous sessions by
   default, lockfile setup, and optional local FlashAttention installation.
2. `scripts/build_flash_attn.sh` plus
   `scripts/patches/flash-attn-2.8.3.post2-sm121.patch` — reproducible SM 12.1
   FA2 build.
3. `tools/benchmark_qwen_tts.py` — repeatable HTTP benchmark.
4. DGX Spark documentation and this report.
5. No change to the native model architecture or the shared server.

The default remains **SDPA**, which is in `uv.lock` and needs no local build.

## Methodology

Benchmarks submitted one job at a time to the HTTP API. Each case used the
1.7B CustomVoice profile, `ryan`, English, the default native sampling settings,
and three runs after one warmup. Wall time is from job submission until the job
reports `succeeded`; authentication/session setup is outside the measured
interval. The benchmark supports anonymous sessions, static bearer tokens via
`--api-key`, and token-free servers. Audio duration is the returned result duration. RTF is wall
time divided by audio duration (lower is better; below 1 is faster than real
time).

The cold-start measurement includes model loading. Warm measurements reuse the
resident model. No Gradio UI or second model was running during the measurements.

## Results

### Warm SDPA baseline

| Case | Audio | Wall | RTF |
| --- | ---: | ---: | ---: |
| 25 chars | 2.32 s | 1.80 s | **0.798** |
| 128 chars | 8.40 s | 6.35 s | **0.749** |
| 211 chars | 12.32 s | 9.38 s | **0.741** |

Raw medians are in `benchmarks/qwen-sdpa.json`.

### Cold start

A cold 7.2 s CustomVoice request completed in **43.3 s**, including about
**37 s** of model loading and the first inference. The resident model used about
**4.5 GiB** in `nvidia-smi`.

### API and MCP

- `/v1/models` advertised the expected 1.7B profile and native controls.
- `/v1/speech` + job polling + `/v1/assets/{id}` returned a valid 24 kHz mono
  WAV.
- MCP at `/mcp` listed `generate_speech`, `get_job`, `list_models`,
  `list_voices`, `clone_voice`, `design_voice` and related tools.
- An MCP `generate_speech` call produced a valid 7.2 s WAV through the same
  jobs/assets pipeline.

### FlashAttention 2 on SM 12.1

| Backend | Case | Audio | Wall | RTF |
| --- | --- | ---: | ---: | ---: |
| SDPA | 25 chars | 2.32 s | 1.80 s | **0.798** |
| FA2 | 25 chars | 2.08 s | 1.80 s | 0.897 |
| SDPA | 128 chars | 8.40 s | 6.35 s | **0.749** |
| FA2 | 128 chars | 9.52 s | 7.59 s | 0.804 |
| SDPA | 211 chars | 12.32 s | 9.38 s | **0.741** |
| FA2 | 211 chars | 15.76 s | 12.89 s | 0.811 |

The wheel built successfully: `flash_attn-2.8.3.post2-cp313-cp313-linux_aarch64.whl`
(63 MB), and `cuobjdump --list-elf` showed **72 SM 12.1 cubins**. The raw
`flash_attn_func` test passed on BF16. The complete model also ran with
`native-12hz/cuda:0/bfloat16/flash_attention_2`.

The end-to-end result is clear: FA2 is **8–12% slower in RTF** than SDPA on
this TTS workload, even though both produce valid audio. The audio durations
differ because native sampling is stochastic; RTF normalizes for that. A
deterministic greedy follow-up was abandoned because this model can fail to emit
EOS under greedy decoding and run until the token cap. The practical decision is
to keep **SDPA as the default** and expose FA2 as an opt-in compatibility path.

The build itself is reproducible but not fast: with `MAX_JOBS=4` and
`NVCC_THREADS=2`, it took about **19 minutes** on the Spark. A plain
high-parallelism build is not recommended; it can starve the ARM cores.


## Analysis

The 1.7B Qwen3-TTS model is an autoregressive speech model with a 12 Hz
tokenizer. At these clip lengths the measured cost is already below real time,
so the headroom is limited by model architecture and sampling rather than by a
single obvious kernel. SDPA uses PyTorch's native fused attention path and is a
strong default on GB10.

FlashAttention is a compatibility build on SM 12.1, not an officially optimized
Blackwell target. The upstream issue reports FA2 kernels compiling and running
on GB10 while tying or losing to SDPA at short sequence lengths, and a GB10
tuning guide independently recommends SDPA over building FA2 for attention-bound
workloads. This fork therefore treats FA2 as an opt-in experiment and keeps SDPA
as the deployment default unless the measured TTS result says otherwise.

The dominant avoidable latency is cold model loading (~37 s). The shared server
loads lazily and has no preload option; the first request pays the cost. Warm
generation is already 0.74–0.80 RTF.

## Limitations

- One model is resident at a time; switching profiles reloads the checkpoint.
- The API returns complete audio containers; there is no incremental audio.
- Cancellation is checked around the native call, not inside the GPU kernel.
- The 1.7B measurements use one voice and one language; they are not a
  multilingual or concurrency benchmark.
- No formal listening/quality evaluation was performed. The WAVs were checked
  for duration, sample rate, channels and successful decoding.
- The SM 12.1 PyTorch warning remains; BF16/SDPA and this model run.
- FlashAttention results are specific to the locally built wheel.

## Reproduce

```bash
cd /home/spark/opt/TTS/dgxspark.qwen3TTS
./start.sh

.venv/bin/python tools/benchmark_qwen_tts.py \
  --url http://127.0.0.1:8000 --runs 3 --output benchmarks/qwen-sdpa.json
```

For the optional FA2 path:

```bash
./scripts/build_flash_attn.sh
QWEN_TTS_ATTENTION=flash_attention_2 ./start.sh
.venv/bin/python tools/benchmark_qwen_tts.py \
  --url http://127.0.0.1:8000 --runs 3 --output benchmarks/qwen-fa2.json
```

## Files

| File | Purpose |
| --- | --- |
| `start.sh` | HTTP + MCP launcher. |
| `scripts/build_flash_attn.sh` | SM 12.1 FA2 build. |
| `scripts/patches/flash-attn-2.8.3.post2-sm121.patch` | CUDA arch patch. |
| `tools/benchmark_qwen_tts.py` | HTTP benchmark. |
| `docs/dgx-spark.md` | Operating guide. |
| `docs/flash-attention.md` | Build and verification details. |
| `benchmarks/qwen-sdpa.json` | SDPA samples. |
| `benchmarks/qwen-fa2.json` | FA2 samples. |
