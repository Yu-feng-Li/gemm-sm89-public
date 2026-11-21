#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
cuda_root=${CUDA_ROOT:-${CUDA_HOME:-}}
if [[ -z "$cuda_root" ]]; then
    if compiler=$(command -v nvcc); then
        cuda_root=$(dirname -- "$(dirname -- "$(readlink -f "$compiler")")")
    else
        cuda_root=/usr/local/cuda
    fi
fi
cuda_root=$(realpath -- "$cuda_root")
if [[ ! -x "$cuda_root/bin/nvcc" ]]; then
    echo 'CUDA Toolkit nvcc is required. Set CUDA_ROOT to its installation directory.' >&2
    exit 1
fi
cutlass=third_party/cutlass
pin=2c85f497547eb441f0fe80bfb0fb2e6f9add1719
if [[ ! -e "$cutlass" ]]; then
    mkdir -p third_party
    checkout=$(mktemp -d third_party/.cutlass-XXXXXX)
    git -C "$checkout" init -q
    git -C "$checkout" fetch --depth=1 https://github.com/NVIDIA/cutlass.git "$pin"
    git -C "$checkout" checkout --quiet --detach FETCH_HEAD
    mv -T -- "$checkout" "$cutlass"
fi
if [[ ! -d "$cutlass/.git" || "$(git -C "$cutlass" rev-parse HEAD)" != "$pin" ]] ||
   [[ -n "$(git -C "$cutlass" status --short -- include)" ]]; then
    echo "Expected unmodified CUTLASS headers at $pin." >&2
    exit 1
fi
mkdir -p build
"$cuda_root/bin/nvcc" -std=c++17 -O3 -shared -Xcompiler=-fPIC -arch=sm_89 \
    -I "$cutlass/include" bench/tf32_backends.cu -o build/libtf32_backends.so \
    -L "$cuda_root/lib64" -lcublas -lcublasLt -Xlinker -rpath -Xlinker "$cuda_root/lib64"
printf '%s\n' "$cuda_root" > build/cuda-root.txt
