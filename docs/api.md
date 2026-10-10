# Shared Qwen TTS API on DGX Spark

This integration supplies native Qwen adapters to
[python.tts.api.server](https://github.com/lambdawalker/python.tts.api.server).
The server owns HTTP, optional MCP, authentication, uploads, voices, SQLite jobs,
idempotency, cancellation and reconnectable SSE. This repository owns model
loading and native inference. The existing `qwen-tts-demo` Gradio UI stays available.

The [central TTS design](https://github.com/lambdawalker/design.ai/tree/main/tts)
is the architecture authority. Use the
[Python client](https://github.com/lambdawalker/python.tts.api.client) from any
client machine; it does not need Torch or CUDA.

## Install and launch

Run on **Linux ARM64 DGX Spark, Python 3.13, CUDA 13-compatible driver**:

```bash
uv sync --frozen --extra api
export TTS_API_TOKENS='{"replace-with-a-long-random-token":"my-client"}'
uv run --frozen --extra api qwen-tts-api --port 8000 --data-dir .tts-data
```

`uv.lock` pins the deployment, including Torch 2.10.0+cu130 and shared server
commit `9257bc16e60be92e8418455453c3f00bd6ac0f13`. A PyPI release of the shared
server is not a prerequisite. Use `uv` so its Git source mapping is honored;
plain `pip install .[api]` does not apply `[tool.uv.sources]`.

Default inference is `cuda:0`, `bfloat16`, **SDPA**. FlashAttention is optional and
is not installed or assumed. Models load lazily, with only one model resident at
a time. First inference downloads the selected immutable Hub snapshot, including
its bundled tokenizer. Reserve disk and memory for your enabled checkpoints.
Discovery does not load weights or require network access with the shipped pins.
A profile marked `loadable` is not proof of successful GPU inference.

The default bind address is `0.0.0.0` (all IPv4 interfaces). From another machine,
connect to `http://<spark-lan-ip>:8000`, not `0.0.0.0`. Network binding requires
`TTS_API_TOKENS` unless you explicitly pass `--anonymous-sessions` or `--no-auth`. To restrict access to this
machine and optionally run without a token, explicitly pass `--host 127.0.0.1`.

To enable MCP as well:

```bash
export TTS_API_TOKENS='{"replace-with-a-long-random-token":"my-client"}'
uv run --frozen --extra api qwen-tts-api --port 8000 --mcp --data-dir .tts-data
```

Use a trusted network or TLS reverse proxy. Clients send `Authorization: Bearer ...`.
The same authentication protects `/mcp`, SSE and asset downloads. Never commit tokens.
Do not run multiple workers/processes against the same data directory. Gradio is a
separate application; avoid serving it concurrently on the same GPU without a memory budget.

`qwen-tts-api --help` exposes the shared deployment limits, default profile,
retention and queue settings. Equivalently:

```bash
uv run --frozen --extra api tts-api-server --adapter qwen_tts_api.factory:create_adapter --host 0.0.0.0 --mcp
```

## Profiles and controls

| Profile ID | Operation | Voice input | Instructions |
| --- | --- | --- | --- |
| `qwen3-tts-1.7b-customvoice` (default) | Speech | Explicit preset ID or alias | Supported |
| `qwen3-tts-0.6b-customvoice` | Speech | Explicit preset ID or alias | Rejected |
| `qwen3-tts-1.7b-base` | Speech cloning | One registered reference | Rejected |
| `qwen3-tts-0.6b-base` | Speech cloning | One registered reference | Rejected |
| `qwen3-tts-1.7b-voicedesign` | Voice design | Description plus preview text | Description maps to native `instruct` |

Voice conversion is unsupported. VoiceDesign is submitted through
`POST /v1/voice-designs`, not the speech endpoint. Its candidate can be explicitly
registered against a Base profile with the preview transcript, then used for new speech.
CustomVoice aliases: `vivian`, `serena`, `uncle_fu`, `dylan`, `eric`, `ryan`, `aiden`,
`ono_anna`, `sohee`. There is no silent default-speaker substitution.

Languages: `auto`, `zh`, `en`, `ja`, `ko`, `de`, `fr`, `ru`, `pt`, `es`, `it`.
Omitting language selects native `Auto`. Outputs are 24 kHz mono WAV or FLAC,
PCM 16-bit. Other requested sample rates are rejected rather than resampled.

Discover effective controls via `/v1/models`, `/v1/capabilities?model=...`,
`/v1/guidance/tts?model=...`, and `/v1/voices?model=...`.
Namespaced `extensions.qwen` accepts:

- `do_sample`, `top_k`, `top_p`, `temperature`, `repetition_penalty`, `max_new_tokens`.
- `subtalker_dosample`, `subtalker_top_k`, `subtalker_top_p`, `subtalker_temperature`.
- `non_streaming_mode` (a native generation mode; it does **not** enable streamed HTTP audio).
- Base only: `x_vector_only_mode`, default false.

Schemas report bounds and reject unknown controls, including `seed`. Native generation
defaults are resolved from the pinned checkpoint and recorded in result provenance.
The deployment caps text/description at 10,000 characters, references at one clip of
60 seconds, and an explicit `max_new_tokens` at 8,192. These are service limits,
not a guarantee that every input fits GPU memory. Native `max_new_tokens` can truncate
speech; callers should choose appropriate text length and listen to results.

Base ICL mode requires a nonempty, accurate reference transcript at synthesis.
Registration permits an absent transcript so it can be used with
`{"qwen":{"x_vector_only_mode":true}}`. This speaker-embedding-only mode may reduce
cloning quality. No ASR fills in missing transcripts. References are re-encoded per
job; there is no persistent embedding cache in this initial integration.

Text, tags, instructions, descriptions and transcripts are passed through unchanged.
Qwen applies its own chat templates and tokenization. Uploaded reference audio is
decoded to float32 and channels averaged to mono; native speaker and tokenizer encoders
resample as required. These technical operations are declared in result provenance.
No denoising, trimming, translation or tag rewriting is performed.

## Client example and hardware smoke test

Start the server as above, then from this checkout on a client machine:

```bash
export TTS_API_KEY='replace-with-the-server-token' # omit for local unauthenticated mode
uv run --no-project --with 'git+https://github.com/lambdawalker/python.tts.api.client@f82cd3e2546e58eb419a9e0e8a79ab398fc6a24b' python examples/api_client.py --url http://127.0.0.1:8000
```

The example writes `api-output/custom.wav`, `design.wav`, and `clone.wav`. It tests
all three 1.7B inference families and profile switching. Listen to the outputs to
confirm speech completeness, language, instruction adherence and cloning quality.
Repeat speech/cloning with the 0.6B IDs when those profiles are deployed. The example
uses three separate jobs; model switching entails unloading/reloading weights.

Minimal HTTP submission:

```bash
curl -H 'Content-Type: application/json' -H "Authorization: Bearer $TTS_API_KEY" \
  -d '{"text":"Hello!","voice":{"alias":"ryan"},"language":"en"}' \
  http://127.0.0.1:8000/v1/speech
```

Use the returned `status_url` to poll or `events_url` for reconnectable SSE; add your
bearer header for authenticated deployments. Download the result's `asset_id` from
`/v1/assets/{id}`. See the shared server's
[wire contract](https://github.com/lambdawalker/python.tts.api.server/blob/main/docs/contracts.md)
and [deployment guide](https://github.com/lambdawalker/python.tts.api.server/blob/main/docs/deployment.md)
for upload, retention, idempotency, SSE replay and authorization semantics.

## Configuration and reproducibility

| Environment variable | Default / meaning |
| --- | --- |
| `QWEN_TTS_PROFILES` | All five; comma-separated IDs. The first is the default unless `--default-model` overrides it. |
| `QWEN_TTS_REVISIONS` | JSON mapping of IDs to Hub revisions; defaults to the immutable commits in `factory.py`. Named refs are resolved before serving. |
| `QWEN_TTS_DEVICE` | `cuda:0` |
| `QWEN_TTS_DTYPE` | `bfloat16`; also `float16` or `float32` |
| `QWEN_TTS_ATTENTION` | `sdpa`; also `eager` or explicitly installed `flash_attention_2` |
| `TTS_API_TOKENS` | Shared server token-to-caller JSON map |
| `HF_HOME` / `HF_HUB_OFFLINE` | Standard Hugging Face cache/offline settings |

For offline use, cache the full pinned snapshots before setting `HF_HUB_OFFLINE=1`.
Revision hashes avoid Hub metadata calls. Unsupported/custom checkpoint families
are not silently accepted: the loaded size, family, languages and preset speakers
must match the advertised profile. Revisions are recorded in discovery and job results.

Progress reports lifecycle stages, not invented completion percentages. Results are
complete audio containers; the integration advertises no incremental audio. Cancellation
is checked before and after the blocking native call, so cancellation can remain pending
while the GPU is busy. Process crashes produce failed interrupted jobs through the
shared server; they do not silently rerun inference.

## Validation and development

CPU CI isolates API checks from the ARM64 CUDA environment:

```bash
uv venv --python 3.13 .venv-api
uv pip sync --python .venv-api/bin/python tests/api/requirements.txt
.venv-api/bin/python -m pytest tests/api -q
.venv-api/bin/ruff check qwen_tts_api tests/api examples/api_client.py
.venv-api/bin/ruff format --check qwen_tts_api tests/api examples/api_client.py
uv lock --check
uv build
```

Tests use a native model double returning silence; they verify mappings, validation,
audio encoding, the existing Python client over HTTP, MCP, profile switching and
cancellation. **They do not validate DGX Spark/CUDA execution or acoustic quality.**
Run the hardware example on the target before claiming deployment readiness.

## Token-free LAN mode

To let clients call the API without a token:

```bash
uv run --frozen --extra api qwen-tts-api --no-auth --mcp
```

The default host is `0.0.0.0`. Connect to `http://<spark-lan-ip>:8000`; omit `api_key`
in the Python client or the Authorization header in HTTP requests. `--no-auth`
overrides `TTS_API_TOKENS` even when it is set. All clients share the `local` caller,
including its jobs, voices and assets. Anyone who can reach the port can access
these resources; this flag does not impose a private-network IP restriction.
Remove `--no-auth` to restore the normal token requirement for network binding.


## Anonymous sessions with private jobs

Anyone who can reach the server can create a session without a preconfigured API key:

```bash
unset TTS_API_TOKENS
uv run --frozen --extra api qwen-tts-api --anonymous-sessions --mcp
```

This mode keeps each session's jobs, uploaded assets and registered voices separate.
It is mutually exclusive with `--no-auth` and static `TTS_API_TOKENS`. The server
still binds to `0.0.0.0`; connect using the Spark's LAN IP.

```python
from tts_api_client import TTSClient

with TTSClient("http://<spark-lan-ip>:8000", anonymous_session=True) as client:
    session = client.create_session()
    # Keep session.access_token private if you need to reconnect later.
    job = client.generate_speech(text="Hello!", voice={"alias": "ryan"})
    result = job.wait(timeout=1800)

with TTSClient("http://<spark-lan-ip>:8000", session_token=session.access_token) as client:
    print(client.get_job(job.id).status)
    # client.revoke_session() explicitly ends access; close() does not.
```

Without explicit `create_session()`, the first operation creates one automatically.
The async client supports the same options. To run the full hardware example,
unset `TTS_API_KEY` and pass `--anonymous-session`; set `TTS_SESSION_TOKEN` to
resume a saved session instead. Tokens are never printed by the example.

Raw HTTP clients first call `POST /v1/sessions`, then send the returned
`access_token` as `Authorization: Bearer ...` for all other endpoints, including
MCP, job events and audio downloads. `DELETE /v1/sessions/current` revokes it.
Tokens expire after 24 hours by default (`--session-ttl` seconds); at most 10,000
active sessions are allowed (`--max-sessions`). Sessions survive server restarts.
There is no automatic renewal or recovery for lost/expired tokens. Expiration or
revocation closes job-event streams but does not cancel jobs or delete resources.

This is session ownership, not user identity: anyone holding a token has its access,
and anyone reaching the port can create a new session and submit work. Use HTTPS
when the network is not trusted. See the
[session contract](https://github.com/lambdawalker/design.ai/blob/main/tts/sessions.md)
for lifecycle, capacity and retention semantics.
