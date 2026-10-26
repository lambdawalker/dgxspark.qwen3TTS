#!/bin/sh
# One-command DGX Spark launcher for the shared Qwen TTS HTTP/MCP server.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd -- "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi
if [ "$(uname -m)" != "aarch64" ]; then
    echo "start.sh targets Linux ARM64 (DGX Spark); got $(uname -m)." >&2
    exit 1
fi

PY="$ROOT/.venv/bin/python"
# Reconcile locked dependencies on every start, including after git pull.
# Keep extra packages such as the locally built FlashAttention wheel.
echo "Synchronizing the locked Qwen TTS API environment..."
uv sync --frozen --extra api --inexact

if [ "${QWEN_TTS_ATTENTION:-sdpa}" = "flash_attention_2" ]; then
    if ! "$PY" -c 'import flash_attn' >/dev/null 2>&1; then
        WHEEL=$(ls -1t "$ROOT"/wheels/flash_attn-*.whl 2>/dev/null | head -1 || true)
        if [ -z "$WHEEL" ]; then
            echo "FlashAttention was selected but is not installed and no local wheel exists." >&2
            echo "Run: ./scripts/build_flash_attn.sh" >&2
            exit 1
        fi
        echo "Installing FlashAttention wheel: $WHEEL"
        uv pip install --python "$PY" --no-deps "$WHEEL"
    fi
fi

HOST="${QWEN_TTS_HOST:-0.0.0.0}"
PORT="${QWEN_TTS_PORT:-8000}"
DATA_DIR="${QWEN_TTS_DATA_DIR:-.tts-data}"
MCP_ARGS=""
if [ "${QWEN_TTS_MCP:-1}" != "0" ]; then
    MCP_ARGS="--mcp"
fi
AUTH_ARGS=""
case "${QWEN_TTS_AUTH:-auto}" in
    auto)
        if [ -z "${TTS_API_TOKENS:-}" ]; then
            AUTH_ARGS="--anonymous-sessions"
        fi
        ;;
    token) ;;
    anonymous) AUTH_ARGS="--anonymous-sessions" ;;
    none) AUTH_ARGS="--no-auth" ;;
    *)
        echo "QWEN_TTS_AUTH must be auto, token, anonymous or none" >&2
        exit 1
        ;;
esac

echo "Starting Qwen TTS API on http://$HOST:$PORT"
echo "Data directory: $DATA_DIR; MCP: $([ -n "$MCP_ARGS" ] && echo on || echo off)"
# shellcheck disable=SC2086
exec "$ROOT/.venv/bin/qwen-tts-api" \
    --host "$HOST" \
    --port "$PORT" \
    --data-dir "$DATA_DIR" \
    $AUTH_ARGS $MCP_ARGS "$@"
