"""Shared device buffers, reference checks and CUDA Graph timing."""

import ctypes as ct
import fcntl
from dataclasses import dataclass
import json
from math import ceil
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time

import numpy as np

from gpu import checked, cuda
from tf32_api import Gemm

ROOT = Path(__file__).resolve().parents[1]
MIN_MS = 10.0
SMALL_REFERENCE = 256**3


def log(path, data):
    with path.open("a") as f:
        f.write(json.dumps(data, allow_nan=False) + "\n")


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


class Backends:
    def __init__(self, shape, scratch, budget, stream, life):
        library = ROOT / "build/libtf32_backends.so"
        if not library.is_file():
            raise FileNotFoundError("library baselines require bash scripts/build.sh")
        self.lib = ct.CDLL(str(library))
        p, i, z = ct.c_void_p, ct.c_int, ct.c_size_t
        declarations = (
            ("tf32_backends_create", p, [i, i, i, p, z, p, ct.POINTER(i)]),
            ("tf32_backends_destroy", None, [p]),
            ("tf32_cublas_version", i, [p, ct.POINTER(i)]),
            ("tf32_lt_version", z, []),
            ("tf32_cublas_status", ct.c_char_p, [i]),
            ("tf32_cutlass_status", ct.c_char_p, [i]),
            ("tf32_cublas", i, [p, p, p, p, i]),
            ("tf32_lt_count", i, [p, ct.POINTER(i)]),
            ("tf32_lt_metadata", i, [p, i, ct.POINTER(i), ct.POINTER(z)]),
            ("tf32_lt", i, [p, i, p, p, p]),
            ("tf32_cutlass", i, [p, i, i, p, p, p, ct.POINTER(z), i]),
        )
        for name, returns, args in declarations:
            fn = getattr(self.lib, name)
            fn.restype, fn.argtypes = returns, args
        err = i()
        self.handle = self.lib.tf32_backends_create(*shape, int(scratch), budget, int(stream), ct.byref(err))
        self.check_blas(err.value, "create library backends")
        if not self.handle:
            raise RuntimeError("missing library handle")
        life.callback(self.lib.tf32_backends_destroy, self.handle)

    def cublas(self, a, b, c, pedantic=0):
        name = "cuBLAS FP32_PEDANTIC" if pedantic else "cuBLAS FAST_TF32"
        self.check_blas(self.lib.tf32_cublas(self.handle, int(a), int(b), int(c), pedantic), name)

    def check_blas(self, code, operation):
        if code:
            reason = self.lib.tf32_cublas_status(code).decode()
            raise RuntimeError(f"{operation}: {reason} ({code})")

    def versions(self):
        version = ct.c_int()
        self.check_blas(self.lib.tf32_cublas_version(self.handle, ct.byref(version)), "cuBLAS version")
        return dict(cublas_api=version.value, cublaslt_api=self.lib.tf32_lt_version())

    def lt_catalog(self):
        err = ct.c_int()
        count = self.lib.tf32_lt_count(self.handle, ct.byref(err))
        names = ("algo_id", "tile_id", "split", "reduction", "stages", "swizzle")
        rows = []
        for index in range(count):
            fields, work = (ct.c_int * len(names))(), ct.c_size_t()
            rc = self.lib.tf32_lt_metadata(self.handle, index, fields, ct.byref(work))
            rows.append(dict(index=index, status=rc, **dict(zip(names, fields))))
        return dict(
            returned=count,
            heuristic_status=err.value,
            reason=self.lib.tf32_cublas_status(err.value).decode(),
            algorithms=rows,
        )

    def lt(self, index, a, b, c):
        self.check_blas(
            self.lib.tf32_lt(self.handle, index, int(a), int(b), int(c)), f"cuBLASLt heuristic index {index}"
        )

    def cutlass(self, tile, split, a, b, c, check=False):
        work = ct.c_size_t()
        rc = self.lib.tf32_cutlass(
            self.handle, tile, split, int(a), int(b), int(c), ct.byref(work), int(check)
        )
        if rc:
            reason = "workspace exceeds budget" if rc == -2 else self.lib.tf32_cutlass_status(rc).decode()
            raise RuntimeError(f"CUTLASS tile={tile} split={split}: {reason} ({rc})")


@dataclass
class Method:
    label: str
    backend: str
    metadata: dict
    launch: object
    graph: object = None
    launches_per_graph: int = 0
    graph_repeats: int = 0


def idle(gpu):
    if Clocks(gpu).processes() - {os.getpid()}:
        raise RuntimeError("The selected device already has a compute process")


def gpu_lock(gpu):
    # This lock serializes benchmark processes even across local clones.
    directory = Path(tempfile.gettempdir()) / f"gemm-sm89-{os.getuid()}"
    directory.mkdir(mode=0o700, exist_ok=True)
    lock = (directory / f"{gpu}.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("A benchmark already holds the selected device lock") from None
    return lock


def cuda_toolkit():
    marker = ROOT / "build/cuda-root.txt"
    if not marker.is_file():
        raise RuntimeError("Run bash scripts/build.sh to locate the CUDA toolkit")
    return Path(marker.read_text().strip())


class ComputeProcess(ct.Structure):
    _fields_ = [
        ("pid", ct.c_uint),
        ("used_memory", ct.c_ulonglong),
        ("gpu_instance", ct.c_uint),
        ("compute_instance", ct.c_uint),
    ]


class Clocks:
    def __init__(self, gpu):
        self.nvml = ct.CDLL("libnvidia-ml.so.1")
        self.handle = ct.c_void_p()
        self.call("nvmlInit_v2")
        self.call("nvmlDeviceGetHandleByPciBusId_v2", gpu.encode(), ct.byref(self.handle))

    def call(self, name, *args):
        result = getattr(self.nvml, name)(*args)
        if result:
            raise RuntimeError(f"{name} failed with NVML status {result}")

    def read(self):
        sm, mem = ct.c_uint(), ct.c_uint()
        self.call("nvmlDeviceGetClockInfo", self.handle, ct.c_uint(1), ct.byref(sm))
        self.call("nvmlDeviceGetClockInfo", self.handle, ct.c_uint(2), ct.byref(mem))
        return time.monotonic_ns(), sm.value, mem.value

    def processes(self):
        entries = (ComputeProcess * 64)()
        count = ct.c_uint(len(entries))
        self.call("nvmlDeviceGetComputeRunningProcesses_v2", self.handle, ct.byref(count), entries)
        return {row.pid for row in entries[: count.value]}

    def driver_version(self):
        value = ct.create_string_buffer(80)
        self.call("nvmlSystemGetDriverVersion", value, ct.c_uint(len(value)))
        return value.value.decode()


class Monitor:
    def __init__(self, gpu):
        self.clocks = Clocks(gpu)
        self.rows, self.health, self.errors, self.foreign = [], [], [], set()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.watch)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()

    def watch(self):
        next_process_check = 0
        try:
            while not self.stop.is_set():
                row = self.clocks.read()
                self.rows.append(row)
                if row[0] >= next_process_check:
                    self.foreign.update(self.clocks.processes() - {os.getpid()})
                    temp, power = ct.c_uint(), ct.c_uint()
                    self.clocks.call(
                        "nvmlDeviceGetTemperature", self.clocks.handle, ct.c_uint(0), ct.byref(temp)
                    )
                    self.clocks.call("nvmlDeviceGetPowerUsage", self.clocks.handle, ct.byref(power))
                    self.health.append((row[0], temp.value, power.value))
                    next_process_check = row[0] + 50_000_000
                self.stop.wait(0.002)
        except Exception as error:
            self.errors.append(str(error))

    def partition(self, measured):
        from bisect import bisect_left

        final = max(r["end_ns"] for r in measured.values())
        while (not self.rows or self.rows[-1][0] < final) and time.monotonic_ns() - final < 8_000_000:
            time.sleep(0.001)
        rows = list(self.rows)
        clocks = set()
        coverage = True
        for r in measured.values():
            first = bisect_left(rows, (r["begin_ns"],))
            last = bisect_left(rows, (r["end_ns"],))
            window = rows[max(0, first - 1) : last + 1]
            r["clock_samples"] = window
            clocks.update((sm, mem) for _, sm, mem in window)
            coverage &= (
                first > 0
                and last < len(rows)
                and last - first >= 3
                and all(b[0] - a[0] <= 8_000_000 for a, b in zip(window, window[1:]))
            )
        if self.errors or self.foreign:
            raise RuntimeError(f"NVML monitor errors={self.errors}, foreign compute processes={self.foreign}")
        return sorted(clocks), coverage


def round_tf32(x):
    bits = x.view(np.uint32)
    return (np.where(np.isfinite(x), bits + np.uint32(0x1000), bits) & np.uint32(0xFFFFE000)).view(np.float32)


class Experiment:
    def __init__(self, shape, cache, out, device, life):
        self.shape, self.cache, self.out, self.life = shape, cache, out, life
        self.m, self.n, self.k = shape
        self.long_call = self.m * self.n * self.k >= 4096**3
        self.l2 = checked(
            cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device)
        )
        self.weight_bytes = 4 * self.k * self.n
        self.weight_count = 1 if cache == "hot" else max(2, ceil(4 * self.l2 / self.weight_bytes))
        self.stream = checked(cuda.cuStreamCreate(0))
        life.callback(lambda: checked(cuda.cuStreamDestroy(self.stream)))
        self.a = self.allocate(4 * self.m * self.k)
        self.b = self.allocate(self.weight_bytes * self.weight_count)
        self.c = self.allocate(4 * self.m * self.n)

        self.weights = [
            cuda.CUdeviceptr(int(self.b) + i * self.weight_bytes) for i in range(self.weight_count)
        ]
        self.backend = None
        self.backend_error = None
        try:
            library_bytes = 32 << 20
            self.backend = Backends(shape, self.allocate(library_bytes), library_bytes, self.stream, life)
        except (OSError, AttributeError, RuntimeError) as error:
            self.backend_error = str(error)
        self.first = checked(cuda.cuEventCreate(0))
        self.last = checked(cuda.cuEventCreate(0))
        life.callback(lambda: checked(cuda.cuEventDestroy(self.last)))
        life.callback(lambda: checked(cuda.cuEventDestroy(self.first)))

    def allocate(self, bytes_):
        ptr = checked(cuda.cuMemAlloc(bytes_))
        self.life.callback(lambda: checked(cuda.cuMemFree(ptr)))
        return ptr

    def inputs(self, seed):
        rng = np.random.default_rng(seed)
        self.ha = rng.random((self.m, self.k), dtype=np.float32) - np.float32(0.5)
        checked(cuda.cuMemcpyHtoD(self.a, self.ha.ctypes.data, self.ha.nbytes))
        self.ref = {}
        for j, p in enumerate(self.weights):
            b = rng.random((self.k, self.n), dtype=np.float32) - np.float32(0.5)
            checked(cuda.cuMemcpyHtoD(p, b.ctypes.data, b.nbytes))
            if j in (0, len(self.weights) - 1):
                i, q = np.divmod(self.reference_indices(seed), self.n)
                fa, fb = self.ha[i].astype(np.float64), b[:, q].T.astype(np.float64)
                exact = np.einsum("ij,ij->i", fa, fb)
                scale = np.einsum("ij,ij->i", abs(fa), abs(fb))
                qa = round_tf32(self.ha[i]).astype(np.float64)
                qb = round_tf32(b[:, q].T.copy()).astype(np.float64)
                tf32 = np.einsum("ij,ij->i", qa, qb)
                full = None
                if self.backend is None or self.m * self.n * self.k <= SMALL_REFERENCE:
                    full = self.ha.astype(np.float64) @ b.astype(np.float64)
                if self.backend is None:
                    reference = "cpu_fp64"
                    failures = np.count_nonzero(np.abs(exact - full[i, q]) > 1e-4 + 1e-5 * abs(exact))
                    max_abs = float(np.max(np.abs(exact - full[i, q])))
                else:
                    self.backend.cublas(self.a, p, self.c, 1)
                    pedantic = self.result()
                    reference = "cpu_fp64" if full is not None else "cublas_fp32_pedantic"
                    failures = np.count_nonzero(np.abs(exact - pedantic[i, q]) > 1e-4 + 1e-5 * abs(exact))
                    max_abs = float(np.max(np.abs(exact - pedantic[i, q])))
                    if full is not None and not np.all(np.abs(full - pedantic) <= 1e-4 + 1e-5 * np.abs(full)):
                        raise RuntimeError("FP32 pedantic reference failed full CPU FP64 check")
                log(
                    self.out / "reference.jsonl",
                    dict(
                        seed=seed,
                        ring_index=j,
                        cpu_fp64_samples=len(i),
                        reference=reference,
                        cpu_fp64_failed=int(failures),
                        cpu_max_abs=max_abs,
                    ),
                )
                if failures:
                    raise RuntimeError(f"{reference} reference failed independent FP64 spot check")
                self.ref[j] = (full if full is not None else pedantic, i, q, tf32, scale)
        self.seed = seed

    def reference_indices(self, seed):
        m, n = self.m, self.n
        mandatory = [0, n - 1, (m - 1) * n, m * n - 1]
        choice = np.random.default_rng(seed + 8173).choice(m * n, size=min(m * n, 256), replace=False)
        return np.unique(np.concatenate((mandatory, choice)))

    def result(self):
        checked(cuda.cuStreamSynchronize(self.stream))
        arr = np.empty((self.m, self.n), dtype=np.float32)
        checked(cuda.cuMemcpyDtoH(arr.ctypes.data, self.c, arr.nbytes))
        return arr

    def prepare(self, name, backend, life, metadata=None):
        start = time.monotonic()
        if backend == "cute":
            plan = Gemm(self.shape, cache_policy=self.cache, method=name)
            fields = plan.selected

            def submit(a, b, c):
                plan(a, b, c, self.stream)

        elif backend == "cublas":
            self.require_backend()
            fields = dict(api="FAST_TF32")
            submit = self.backend.cublas
        elif backend == "lt":
            self.require_backend()
            fields = metadata

            def submit(a, b, c):
                self.backend.lt(fields["index"], a, b, c)

        elif backend == "cutlass":
            self.require_backend()
            tile, split = metadata["tile_id"], metadata["split"]
            self.backend.cutlass(tile, split, self.a, self.b, self.c, True)
            fields = metadata

            def submit(a, b, c):
                self.backend.cutlass(tile, split, a, b, c)

        elif backend == "triton":
            from tf32_triton import prepare

            submit, cleanup, fields = prepare(
                name, self.shape, self.stream, checked, self.out, cuda_toolkit()
            )
            life.callback(cleanup)
        else:
            raise ValueError(f"unknown backend {backend}")
        return Method(name, backend, dict(fields, prepare_seconds=time.monotonic() - start), submit)

    def require_backend(self):
        if self.backend is None:
            raise RuntimeError(self.backend_error)

    def validate_method(self, method):
        valid = True
        for j, (ref, i, q, tf32, scale) in self.ref.items():
            checked(cuda.cuMemsetD8Async(self.c, 255, 4 * self.m * self.n, self.stream))
            method.launch(self.a, self.weights[j], self.c)
            out = self.result()
            delta = np.abs(out.astype(np.float64) - ref)
            failing = (~np.isfinite(out)) | (delta > 0.02 + 0.003 * abs(ref))
            point = int(np.argmax(np.where(np.isfinite(delta), delta, np.inf)))
            finite = bool(np.all(np.isfinite(out)))
            l2 = (
                float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(ref.ravel()), 1e-300))
                if finite
                else None
            )
            failed = int(failing.sum())
            passed = finite and not failed and l2 <= 0.001
            err = np.abs(out[i, q] - tf32)
            row = dict(
                method=method.label,
                seed=self.seed,
                ring_index=j,
                valid=passed,
                failed=failed,
                max_abs=float(delta.flat[point]) if finite else None,
                relative_l2=l2,
                worst_mn=divmod(point, self.n),
                tf32_fp64_sample_max_abs=float(max(err)) if finite else None,
                sample_max_abs_product_sum=float(max(scale)),
                metadata=method.metadata,
            )
            log(self.out / "validation.jsonl", row)
            valid &= passed
        return valid

    def capture(self, method, life):
        method.launch(self.a, self.b, self.c)
        checked(cuda.cuStreamSynchronize(self.stream))
        count = self.weight_count if self.long_call else self.weight_count * ceil(20 / self.weight_count)
        checked(
            cuda.cuStreamBeginCapture(self.stream, cuda.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_GLOBAL)
        )
        try:
            for i in range(count):
                method.launch(self.a, self.weights[i % self.weight_count], self.c)
        except Exception:
            cuda.cuStreamEndCapture(self.stream)
            raise
        graph = checked(cuda.cuStreamEndCapture(self.stream))
        try:
            method.graph = checked(cuda.cuGraphInstantiate(graph, 0))
        finally:
            checked(cuda.cuGraphDestroy(graph))
        obj = method.graph
        life.callback(lambda: checked(cuda.cuGraphExecDestroy(obj)))
        method.launches_per_graph = count
        # Pilot graph replay calibrates repeat counts outside the ranking.
        # A hot direct launch above already produced the expected C. Poison it
        # on the same stream so a graph that skips output stores fails replay.
        checked(cuda.cuMemsetD8Async(self.c, 255, 4 * self.m * self.n, self.stream))
        calibration = self.event(method, 3)
        # The graph uses the same C and scratch repeatedly and must leave C for the last B.
        ref = self.ref[self.weight_count - 1][0]
        replay = self.result()
        delta = np.abs(replay.astype(np.float64) - ref)
        correct = bool(
            np.all(np.isfinite(replay))
            and np.all(delta <= 0.02 + 0.003 * abs(ref))
            and np.linalg.norm(delta.ravel()) <= 0.001 * max(np.linalg.norm(ref.ravel()), 1e-300)
        )
        log(
            self.out / "graph-validation.jsonl",
            dict(
                method=method.label,
                seed=self.seed,
                ring_index=self.weight_count - 1,
                valid=correct,
                replays=3,
            ),
        )
        if not correct:
            raise ValueError("graph replay failed final-ring numerical contract")
        total = calibration["total_ms"] / 3
        target = 1.25 if self.long_call else 1.5
        method.graph_repeats = max(1, ceil(target * MIN_MS / max(total, 0.01)))

    def warm(self, method):
        for _ in range(ceil(20 / method.launches_per_graph)):
            checked(cuda.cuGraphLaunch(method.graph, self.stream))
        checked(cuda.cuStreamSynchronize(self.stream))

    def event(self, method, repeats=None):
        repeats = repeats or method.graph_repeats
        begin = time.monotonic_ns()
        checked(cuda.cuEventRecord(self.first, self.stream))
        for _ in range(repeats):
            checked(cuda.cuGraphLaunch(method.graph, self.stream))
        checked(cuda.cuEventRecord(self.last, self.stream))
        checked(cuda.cuEventSynchronize(self.last))
        end = time.monotonic_ns()
        elapsed = checked(cuda.cuEventElapsedTime(self.first, self.last))
        calls = repeats * method.launches_per_graph
        return dict(begin_ns=begin, end_ns=end, total_ms=elapsed, calls=calls, us=1000 * elapsed / calls)

    def matched(self, round_id, order, monitor):
        for attempt in range(1, 31):
            # Large Tensor Core graphs can reach the board power limit during repeated attempts.
            # Keep work serialized and give the GPU a short idle interval before each panel.
            if self.long_call:
                time.sleep(0.8)
            measured = {}
            for method in order:
                if self.long_call:
                    checked(cuda.cuGraphLaunch(method.graph, self.stream))
                    checked(cuda.cuStreamSynchronize(self.stream))
                else:
                    self.warm(method)
                measured[method.label] = self.event(method)
            pairs, coverage = monitor.partition(measured)
            enough = all(r["total_ms"] >= MIN_MS for r in measured.values())
            matched = len(pairs) == 1 and coverage and enough
            begin = min(r["begin_ns"] for r in measured.values())
            end = max(r["end_ns"] for r in measured.values())
            telemetry = [r for r in monitor.health if begin <= r[0] <= end]
            record = dict(
                round=round_id,
                attempt=attempt,
                accepted=matched,
                reasons=[
                    reason
                    for ok, reason in (
                        (len(pairs) == 1, "clocks differ"),
                        (coverage, "telemetry coverage"),
                        (enough, "event shorter than 10 ms"),
                    )
                    if not ok
                ],
                clock_pairs=pairs,
                health_samples_ns_c_mw=telemetry,
                seed=self.seed,
                order=[method.label for method in order],
                methods=measured,
            )
            log(self.out / "rounds.jsonl", record)
            if matched:
                return record
            if not enough:
                for m in order:
                    if measured[m.label]["total_ms"] < MIN_MS:
                        m.graph_repeats *= 2
        raise RuntimeError(f"could not obtain a complete frequency-matched round {round_id}")
