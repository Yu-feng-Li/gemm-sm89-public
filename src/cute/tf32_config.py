"""Decode documented method names and JIT-compile the selected CuTe launch."""

TILES = {
    'legacy_auto': None,
    'dsl_wide': None,
    'dsl_square': ((128, 128, 16), (2, 2, 1), 3),
    'dsl_wide_w1x8': ((128, 256, 16), (1, 8, 1), 3),
    'dsl_64x128_w1x4': ((64, 128, 16), (1, 4, 1), 3),
    'dsl_wide_k32': ((128, 256, 32), (2, 4, 1), 2),
    'dsl_128x160_k32_w8': ((128, 160, 32), (4, 2, 1), 2),
    'dsl_128x96': ((128, 96, 16), (2, 2, 1), 3),
    'dsl_16x32': ((16, 32, 16), (1, 1, 1), 3),
    'dsl_32x128': ((32, 128, 16), (1, 2, 1), 3),
    'dsl_32x64': ((32, 64, 16), (1, 2, 1), 3),
}


def candidate(method, shape, split, cache_all, raster_group, schedule, raster_m):
    from gpu import require_sm89

    require_sm89()
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr
    from tf32_gemm import CTA_TILE, STAGES, WARPS, launch_gemm, launch_split_k

    tile, warps, stages = TILES[method] or (CTA_TILE, WARPS, STAGES)
    compiled = None
    args = ()
    last_addresses = None

    def arguments(addresses):
        a, b, c, scratch, stream = addresses
        pointers = tuple(
            make_ptr(cutlass.Float32, ptr, cute.AddressSpace.gmem, assumed_align=16)
            for ptr in (a, b, c)
        )
        if split > 1:
            pointers += (make_ptr(cutlass.Float32, scratch, cute.AddressSpace.gmem, assumed_align=16),)
        return (*pointers, cuda.CUstream(stream))

    def launch(a, b, c, scratch, stream, prepare):
        nonlocal compiled, args, last_addresses
        addresses = (a, b, c, scratch, stream)
        if addresses != last_addresses:
            args = arguments(addresses)
            last_addresses = addresses
        if prepare:
            *pointers, stream = args
            if method == "legacy_auto":
                compiled = cute.compile(launch_gemm, *pointers, shape, stream)
            elif split == 1:
                compiled = cute.compile(
                    launch_gemm, *pointers, shape, stream, tile, warps, stages,
                    cache_all, raster_group, schedule, raster_m,
                )
            else:
                compiled = cute.compile(
                    launch_split_k, *pointers, shape, tile, warps, stages, split,
                    stream, cache_all, raster_group, schedule, raster_m,
                )
        else:
            compiled(*args)

    return launch


def parse_method(method):
    method, _, schedule = method.partition("__")
    schedule = schedule or "default"
    if schedule not in ("default", "copy_after", "copy_after_unmasked", "steady_copy", "static_raster"):
        raise ValueError(f"unknown MMA schedule {schedule}")
    method, _, raster_text = method.partition("_r")
    raster_m = raster_text.endswith("m")
    raster_group = int(raster_text.removesuffix("m")) if raster_text else 0
    cache_all = method.endswith("_ca")
    method, _, split_text = method.removesuffix("_ca").partition("_split")
    split = int(split_text) if split_text else 1
    if method in ("legacy_auto", "cublas") and (split != 1 or cache_all or raster_group or raster_m or schedule != "default"):
        raise ValueError(f"{method} cannot be combined with explicit tuning overrides")
    if method != "cublas" and method not in TILES:
        raise ValueError(f"unknown method {method}")
    return method, split, cache_all, raster_group, schedule, raster_m
