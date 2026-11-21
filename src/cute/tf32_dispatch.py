"""Resource-constrained SM89 dispatch shared by ordinary calls and the benchmark."""
import hashlib
from functools import lru_cache
import json
from pathlib import Path
import os
from operator import index

ROOT = Path(__file__).resolve().parents[2]
TABLE = Path(os.environ.get("GEMM_DISPATCH_TABLE", ROOT / "configs/tf32-dispatch.json")).resolve()
SOURCES = (
    "src/cute/tf32_gemm.py",
    "src/cute/tf32_config.py",
    "src/cute/tf32_dispatch.py",
    "src/cute/gpu.py",
)

LEGACY_OVERRIDES = {
    (4096, 4096, 4096): "dsl_wide_r1__copy_after",
    (1024, 1024, 1024): "dsl_64x128_w1x4__steady_copy",
    (2048, 2048, 2048): "dsl_wide__steady_copy",
    (512, 5120, 17408): "dsl_128x160_k32_w8__copy_after",
    (4096, 4096, 5120): "dsl_wide_k32__static_raster",
    (4096, 4096, 17408): "dsl_wide_k32__static_raster",
    (256, 256, 256): "dsl_32x64",
    (128, 12288, 5120): "dsl_128x96",
}


def validate_shape(shape):
    try:
        shape = tuple(index(dim) for dim in shape)
    except TypeError as error:
        raise ValueError("expected three integer dimensions M, N, K") from error
    if (len(shape) != 3 or min(shape) <= 0 or max(shape) > 2**31-1
            or shape[1] % 4 or shape[2] < 32 or shape[2] % 16):
        raise ValueError("expected MxNxK with positive M, N multiple of 4 and K >= 32 divisible by 16")
    return shape


def describe(method, shape):
    from tf32_config import TILES, parse_method

    if method in ("auto", "legacy_auto"):
        method = LEGACY_OVERRIDES.get(tuple(shape), "dsl_wide")
    name, split, cache_all, raster, schedule, raster_m = parse_method(method)
    if name == "cublas":
        return dict(selected_method=method, tile=None, split=None)
    tile, warps, stages = TILES[name] or ((128, 256, 16), (2, 4, 1), 3)
    bm, bn, bk = tile
    if (split not in (1, 2, 4, 8, 16) or shape[2] % (bk * split)
            or shape[2] // split < 2*bk):
        raise ValueError(f"{method}: K sections must be divisible by BK and contain at least two tiles")
    threads = 32*warps[0]*warps[1]
    from math import gcd

    if (warps[2] != 1 or bm % (16*warps[0]) or bn % (16*warps[1])
            or bm % (threads//(bk//4)) or bk % (threads//(gcd(bn,128)//4))
            or 4*(bm+bn)*bk*stages > 99*1024):
        raise ValueError(f"{method}: incompatible CTA, copy layout or SM89 shared memory")
    if raster < 0:
        raise ValueError(f"{method}: raster group must be nonnegative")
    gm, gn = (shape[0]+bm-1)//bm, (shape[1]+bn-1)//bn
    return dict(selected_method=method, tile=tile, warps=warps, stages=stages,
                split=split, schedule=schedule,
                raster_group=raster, raster_axis="M" if raster_m else "N", cache_all=cache_all,
                output_ctas=gm*gn, compute_ctas=gm*gn*split,
                fill_ratio=shape[0]*shape[1]/(gm*gn*bm*bn))


@lru_cache(maxsize=1)
def fingerprint():
    from importlib.metadata import version

    return dict(sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},
                cutlass_dsl=version("nvidia-cutlass-dsl"), cuda_bindings=version("cuda-bindings"))


def selection(method, shape, cache="hot", *, table_path=TABLE, identity=None):
    from gpu import checked, cuda, device_model

    shape = validate_shape(shape)
    if cache not in ("hot", "rotate"):
        raise ValueError("expected hot or rotate cache policy")
    result = dict(requested_method=method, shape=shape, cache_policy=cache,
                  dispatch_hit=False, evidence=None)
    if method == "auto":
        table_path = Path(table_path)
        table = json.loads(table_path.read_text()) if table_path.exists() else {}
        key = "x".join(map(str, shape))
        entry = table.get("shapes", {}).get(key, {}).get(cache)
        reason = "unseen_shape_or_policy"
        if entry:
            model, driver = identity or (device_model(), checked(cuda.cuDriverGetVersion()))
            if table.get("fingerprint") != fingerprint() or table.get("driver_api") != driver:
                reason = "stale_table"
            elif model != table.get("gpu_model"):
                reason = "unvalidated_model"
            else:
                return dict(result, **describe(entry["method"], shape), dispatch_hit=True,
                            reason="frozen_exact_shape", evidence=entry["evidence"])
        # Four wide output CTAs leave most of a 4090 idle on this small-M family.
        m, n, k = shape
        fallback = "legacy_auto"
        if cache == "hot" and (n, k) == (1024, 5120) and 17 <= m <= 64:
            fallback = "dsl_16x32"
        return dict(result, **describe(fallback, shape), reason=reason)
    config = describe(method, shape)
    if method == "legacy_auto":
        config["selected_method"] = "legacy_auto"
    return dict(result, **config, reason="explicit")
