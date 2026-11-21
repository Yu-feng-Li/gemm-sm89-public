"""Pinned independent SM89 matmul. Builds a cubin and launches on the caller's CUDA stream."""

import ctypes as ct
import re
import subprocess

import triton
import triton.language as tl
from cuda.bindings import driver as cuda
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def round_to_tf32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b32 bits; cvt.rna.tf32.f32 bits, $1; mov.b32 $0, bits; }",
        "=f,f",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def triton_tf32(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    gm, gn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    first_m = pid // (GROUP * gn) * GROUP
    group_m = tl.minimum(gm - first_m, GROUP)
    tile_m = first_m + (pid % (GROUP * gn)) % group_m
    tile_n = (pid % (GROUP * gn)) // group_m
    row = tile_m * BM + tl.arange(0, BM)
    col = tile_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + rk
        a = tl.load(A + row[:, None] * K + k[None, :], (row[:, None] < M) & (k[None, :] < K), other=0)
        b = tl.load(B + k[:, None] * N + col[None, :], (k[:, None] < K) & (col[None, :] < N), other=0)
        # Explicit same rounding as CuTe. This work is inside the timed kernel.
        acc = tl.dot(round_to_tf32(a), round_to_tf32(b), acc, input_precision="tf32", out_dtype=tl.float32)
    tl.store(C + row[:, None] * N + col[None, :], acc, (row[:, None] < M) & (col[None, :] < N))


PRESETS = {
    "triton_default": (64, 64, 32, 4, 3),
    "triton_small": (16, 64, 32, 4, 3),
    "triton_wide": (32, 128, 32, 4, 3),
    "triton_large": (64, 128, 32, 4, 3),
}


def prepare(name, shape, stream, checked, out, cuda_toolkit):
    bm, bn, bk, warps, stages = PRESETS[name]
    constants = dict(zip(("M", "N", "K", "BM", "BN", "BK", "GROUP"), (*shape, bm, bn, bk, 8)))
    signature = dict(A="*fp32", B="*fp32", C="*fp32")
    signature.update((key, "constexpr") for key in constants)
    src = ASTSource(
        triton_tf32,
        signature,
        constexprs=constants,
        attrs={(i,): [["tt.divisibility", 16]] for i in range(3)},
    )
    binary = triton.compile(
        src, target=GPUTarget("cuda", 89, 32), options=dict(num_warps=warps, num_stages=stages)
    )
    ptx = binary.asm["ptx"]
    if "mma.sync" not in ptx or ".tf32.tf32" not in ptx or "cvt.rna.tf32.f32" not in ptx:
        raise ValueError("Triton PTX lacks ordinary TF32 MMA or explicit input rounding")
    (out / f"{name}.ptx").write_text(ptx)
    artifact = out / f"{name}.cubin"
    artifact.write_bytes(binary.asm["cubin"])
    tool = cuda_toolkit / "bin/cuobjdump"
    sass = subprocess.check_output([str(tool), "--dump-sass", str(artifact)], text=True)
    if not re.search(r"HMMA\.[^ ]*TF32", sass):
        raise ValueError("Triton cubin lacks TF32 Tensor instructions")
    # Triton 3.7.1 appends global and profiling scratch pointers to its ABI.
    # This direct launcher only supports kernels that do not require either.
    if binary.metadata.global_scratch_size or binary.metadata.profile_scratch_size:
        raise ValueError("Triton kernel requires auxiliary scratch")
    entry_args = re.search(r"\.entry\s+\w+\s*\((.*?)\)", ptx, re.S)
    if entry_args is None or entry_args[1].count(".param") != 5:
        raise ValueError("unexpected Triton CUDA kernel parameter layout")
    module = checked(cuda.cuModuleLoadData(binary.asm["cubin"]))
    fn = checked(cuda.cuModuleGetFunction(module, binary.metadata.name.encode()))
    shared = binary.metadata.shared
    if shared > 48 * 1024:
        checked(
            cuda.cuFuncSetAttribute(
                fn, cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared
            )
        )
    ctas = triton.cdiv(shape[0], bm) * triton.cdiv(shape[1], bn)
    driver = ct.CDLL("libcuda.so.1")
    raw_launch = driver.cuLaunchKernel
    raw_launch.argtypes = (
        [ct.c_void_p] + [ct.c_uint] * 7 + [ct.c_void_p, ct.POINTER(ct.c_void_p), ct.c_void_p]
    )
    raw_launch.restype = ct.c_int

    def launch(a, b, c):
        values = (ct.c_uint64 * 5)(int(a), int(b), int(c), 0, 0)
        pointers = (ct.c_void_p * 5)(*(ct.addressof(values) + i * 8 for i in range(5)))
        rc = raw_launch(int(fn), ctas, 1, 1, warps * 32, 1, 1, shared, int(stream), pointers, None)
        if rc:
            raise RuntimeError(f"Triton CUDA launch returned status {rc}")

    return (
        launch,
        lambda: checked(cuda.cuModuleUnload(module)),
        dict(tile=(bm, bn, bk), warps=warps, stages=stages, ptx=f"{name}.ptx"),
    )
