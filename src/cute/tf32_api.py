"""Plan a row-major SM89 TF32 GEMM and submit it to a caller-owned CUDA stream."""

from threading import Lock
import time

from gpu import checked, cuda
from tf32_dispatch import selection, validate_shape
from tf32_config import candidate, parse_method


class Gemm:
    """Owns compiled host state. The caller owns the buffers and stream lifetimes."""

    def __init__(self, shape, *, cache_policy="hot", method="auto", strides=None):
        self.shape = validate_shape(shape)
        m, n, k = self.shape
        if strides is not None and tuple(tuple(s) for s in strides) != ((k, 1), (n, 1), (n, 1)):
            raise ValueError("A, B and C must have contiguous row-major strides")
        self.selected = selection(method, self.shape, cache_policy)
        name, split, cache_all, raster, schedule, raster_m = parse_method(self.selected["selected_method"])
        if name == "cublas":
            raise ValueError("Gemm accepts CuTe configurations")
        self._launch = candidate(name, self.shape, split, cache_all, raster, schedule, raster_m)
        self._partial_bytes = 4 * split * m * n if split > 1 else 0
        self._lock = Lock()
        self._compiled = False
        self.compile_seconds = None

    def __call__(self, a, b, c, stream):
        m, n, k = self.shape
        pointers = tuple(map(int, (a, b, c)))
        ranges = list(zip(pointers, (4 * m * k, 4 * k * n, 4 * m * n)))
        if any(address <= 0 or address % 16 for address, _ in ranges):
            raise ValueError("device buffers must be non-null and 16-byte aligned")
        dst, size = ranges[2]
        if any(dst < src + length and src < dst + size for src, length in ranges[:2]):
            raise ValueError("output must be disjoint from inputs")

        with self._lock:
            # Stream-ordered lifetime also works for overlapping calls and CUDA Graph replay.
            partial = checked(cuda.cuMemAllocAsync(self._partial_bytes, stream)) if self._partial_bytes else None
            try:
                args = (*pointers, int(partial) if partial is not None else None, int(stream))
                if not self._compiled:
                    begin = time.monotonic()
                    self._launch(*args, prepare=True)
                    self.compile_seconds = time.monotonic() - begin
                    self._compiled = True
                self._launch(*args, prepare=False)
            finally:
                if partial is not None:
                    checked(cuda.cuMemFreeAsync(partial, stream))
