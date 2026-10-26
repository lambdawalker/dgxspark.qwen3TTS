#!/bin/sh
# Build FlashAttention 2 for the DGX Spark GB10 (SM 12.1).
#
# Official wheels do not cover linux_aarch64 + sm_121. This script downloads the
# 2.8.3.post2 source, applies the sm_121 setup.py patch, and builds a local wheel.
# It is intentionally resource-limited: a bare high-parallelism CUDA build can
# make a GB10 unresponsive.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
VERSION=2.8.3.post2
URL="https://files.pythonhosted.org/packages/6c/b1/b77625e3d11260d2bd6a4fc249a4be9a3da97ff848e2222e810c4f0e0211/flash_attn-${VERSION}.tar.gz"
SHA256=d0f9f0b08a986cdeadbffcf5ec8b3a8f514366c68ad1853d3ab061bc83cc340b
BUILD_ROOT="${FLASH_ATTN_BUILD_DIR:-$ROOT/.build}"
SOURCE_DIR="$BUILD_ROOT/flash_attn-$VERSION"
WHEEL_DIR="${FLASH_ATTN_WHEEL_DIR:-$ROOT/wheels}"
PATCH="$ROOT/scripts/patches/flash-attn-${VERSION}-sm121.patch"
PY="$ROOT/.venv/bin/python"

if [ ! -x "$PY" ]; then
    echo "Create the environment first: uv sync --frozen --extra api" >&2
    exit 1
fi
if [ "$(uname -m)" != aarch64 ]; then
    echo "This build script targets DGX Spark aarch64; got $(uname -m)." >&2
    exit 1
fi
if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc is required." >&2
    exit 1
fi

mkdir -p "$BUILD_ROOT" "$WHEEL_DIR"
if [ ! -f "$BUILD_ROOT/flash_attn.tar.gz" ]; then
    echo "Downloading FlashAttention $VERSION..."
    curl -L --fail --max-time 120 -o "$BUILD_ROOT/flash_attn.tar.gz" "$URL"
fi
echo "$SHA256  $BUILD_ROOT/flash_attn.tar.gz" | sha256sum -c -
if [ ! -d "$SOURCE_DIR" ]; then
    tar -xzf "$BUILD_ROOT/flash_attn.tar.gz" -C "$BUILD_ROOT"
fi
if ! grep -q 'compute_121' "$SOURCE_DIR/setup.py"; then
    (cd "$SOURCE_DIR" && patch -p1 < "$PATCH")
fi

# pip/wheel/ninja are build-only; a later `uv sync` removes them.
uv pip install --python "$PY" pip wheel ninja >/dev/null

MAX_JOBS="${MAX_JOBS:-4}"
NVCC_THREADS="${NVCC_THREADS:-2}"
echo "Building FlashAttention for sm_121 with MAX_JOBS=$MAX_JOBS, NVCC_THREADS=$NVCC_THREADS"
echo "This typically takes tens of minutes and is run at nice 19."
(
    cd "$SOURCE_DIR"
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE \
    FLASH_ATTN_CUDA_ARCHS=121 \
    MAX_JOBS="$MAX_JOBS" \
    NVCC_THREADS="$NVCC_THREADS" \
    CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS" \
    nice -n 19 "$PY" -m pip wheel . --no-build-isolation --no-deps -w "$WHEEL_DIR"
)

WHEEL=$(ls -1t "$WHEEL_DIR"/flash_attn-*.whl 2>/dev/null | head -1 || true)
if [ -z "$WHEEL" ]; then
    echo "No wheel was produced." >&2
    exit 1
fi
echo "Built: $WHEEL"
if [ "${INSTALL_AFTER_BUILD:-0}" = "1" ]; then
    uv pip install --python "$PY" --no-deps "$WHEEL"
    "$PY" -c 'import flash_attn; print("flash_attn", flash_attn.__version__)'
fi
echo "Install with: uv pip install --python .venv/bin/python --no-deps '$WHEEL'"
