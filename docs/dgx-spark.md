# Qwen3-TTS on the DGX Spark

This fork runs the native Qwen3-TTS family behind the shared
[python.tts.api.server](https://github.com/lambdawalker/python.tts.api.server).
The server owns HTTP, MCP, jobs, assets, voice registration and authentication;
`qwen_tts_api` owns model loading and inference.

Validated on **2026-10-26** with:

| Component | Value |
| --- | --- |
| GPU | NVIDIA GB10, SM 12.1, 121 GiB unified memory |
| Driver / CUDA | 580.173.02 / 13.0 |
| Python | 3.13.15 |
| PyTorch | 2.10.0+cu130 |
| Transformers | 4.57.3 |
| Model | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| Runtime | `cuda:0`, BF16, SDPA, 24 kHz mono |

## Quick start

```bash
cd /home/spark/opt/TTS/dgxspark.qwen3TTS
./start.sh
```

`start.sh` synchronizes the locked `.venv` on every launch and starts the HTTP API and MCP
server on `0.0.0.0:8000` using `.tts-data`. By default it uses
`--anonymous-sessions --mcp`, so callers can create isolated sessions without a
preconfigured token. MCP clients must send the session bearer token; for a
token-free local MCP endpoint use `QWEN_TTS_AUTH=none ./start.sh` (this exposes
the port to everyone who can reach it).

OpenAPI/discovery:

```bash
# Anonymous mode: create a session, then use its token for HTTP and MCP.
TOKEN=$(curl --fail -sS -X POST http://127.0.0.1:8000/v1/sessions \
  -H 'Content-Type: application/json' -d '{}' \
  | .venv/bin/python -c 'import json, sys; print(json.load(sys.stdin)["access_token"])')
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/v1/models
```

The MCP endpoint is `http://127.0.0.1:8000/mcp`. It exposes `list_models`,
`get_capabilities`, `get_guidance`, `list_voices`, `clone_voice`,
`delete_voice`, `validate_speech`, `get_job`, `cancel_job`,
`generate_speech`, `design_voice` and `convert_voice`.

Controls:

| Variable | Default | Meaning |
| --- | --- | --- |
| `QWEN_TTS_HOST` | `0.0.0.0` | Bind address. |
| `QWEN_TTS_PORT` | `8000` | HTTP/MCP port. |
| `QWEN_TTS_DATA_DIR` | `.tts-data` | Jobs, assets and voices. |
| `QWEN_TTS_MCP` | `1` | `0` disables the MCP mount. |
| `QWEN_TTS_AUTH` | `auto` | `auto`, `token`, `anonymous`, or `none`. |
| `TTS_API_TOKENS` | unset | Required when `QWEN_TTS_AUTH=token`. |
| `QWEN_TTS_ATTENTION` | `sdpa` | `sdpa`, `eager`, or a locally built `flash_attention_2`. |
| `QWEN_TTS_PROFILES` | all five | Comma-separated profile IDs; the first is the default. |

`QWEN_TTS_AUTH=auto` uses anonymous sessions when `TTS_API_TOKENS` is unset and
bearer tokens when it is set. `none` is the explicit token-free LAN mode and is
not a private-network restriction.

## Warm performance

Measured through the HTTP API with
[`tools/benchmark_qwen_tts.py`](../tools/benchmark_qwen_tts.py), 1.7B
CustomVoice profile, `ryan`, English, three runs per case after a warmup:

| Case | Audio | Wall | RTF |
| --- | ---: | ---: | ---: |
| 25 chars | 2.32 s | 1.80 s | **0.798** |
| 128 chars | 8.40 s | 6.35 s | **0.749** |
| 211 chars | 12.32 s | 9.38 s | **0.741** |

The first inference also loads the model and takes about **37 s** before
generation; a cold 7.2 s clip completed in 43.3 s. The model itself used about
**4.5 GiB** of the unified memory pool. Once resident, the 1.7B model is faster
than real time on the GB10.

The native defaults are `max_new_tokens=8192`, `non_streaming_mode=true`; the
API reports the effective values in `/v1/models` and job provenance. This
integration returns complete audio containers, not incremental audio.

## FlashAttention

The optional FA2 path does not have an official `linux_aarch64 + sm_121` wheel.
[`scripts/build_flash_attn.sh`](../scripts/build_flash_attn.sh) builds
FlashAttention 2.8.3.post2 with an explicit `compute_121/sm_121` target. See
[FlashAttention on the DGX Spark](flash-attention.md) for the patch, build time,
resource limits and verification steps.

The default remains **SDPA**. A local wheel can be selected with:

```bash
QWEN_TTS_ATTENTION=flash_attention_2 ./start.sh
```

Measured on this Spark, the locally built FA2 wheel is **8–12% slower in RTF**
than SDPA (warm 1.7B CustomVoice, same benchmark). The build succeeds and the
kernel runs, but it is a compatibility target rather than a GB10 optimization.
Use SDPA unless you have a workload that specifically benefits from FA2.

Do not run a plain `uv sync` after installing the local wheel: it is not in
`uv.lock`, so uv removes it. `start.sh` uses
`uv sync --frozen --extra api --inexact` on every launch to update locked
dependencies while preserving extra packages, including the local FA2 wheel.
It installs the wheel on demand when the FA2 runtime is selected.

## API / MCP

The full wire contract, profile table, language list, voice registration,
idempotency, SSE and session behavior are in [the API guide](api.md). A minimal
HTTP request:

```bash
# Use TOKEN from the anonymous session above, or your configured bearer token.
curl -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"text":"Hello from the Spark.","voice":{"alias":"ryan"},"language":"en"}' \
  http://127.0.0.1:8000/v1/speech
```

The response is a queued job. Poll `status_url`, then download the result's
`asset_id` from `/v1/assets/{id}`, using the same bearer token for each request.
MCP clients use the same jobs and assets.

The benchmark automatically creates one anonymous session when required and
reuses its token for all jobs and audio downloads. For a server configured with
static tokens, pass `--api-key TOKEN`. Token-free servers also work without
additional flags. Session setup happens before measured generation.

## Notes and limits

- One model is resident at a time; switching profiles unloads and reloads.
- The API is single-process. Do not point two processes at the same data
  directory.
- CPU tests in `tests/api` do not validate CUDA or acoustics. The benchmarks
  above are the GPU validation.
- The SM 12.1 PyTorch warning is expected; BF16/SDPA and this TTS workload run.
- Keep a GPU/memory budget when running Fish Speech, Gradio and Qwen together.
- The 121 GiB pool is shared between CPU and GPU; `nvidia-smi` reports memory as
  `[N/A]` on GB10.
