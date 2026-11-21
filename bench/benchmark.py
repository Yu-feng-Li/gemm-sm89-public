"""Compare TF32 backends on two fixed shapes with identical inputs."""

import argparse
from collections import defaultdict
from contextlib import ExitStack
import csv
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import random
from statistics import fmean, median, stdev
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/cute"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

from gpu import checked, cuda, current_device, primary_context
from harness import Clocks, Experiment, MIN_MS, Monitor, cuda_toolkit, gpu_lock, idle, log, save
from tf32_dispatch import TABLE, fingerprint

BACKENDS = ("cute", "cublas", "cublaslt", "cutlass", "triton")
SCREEN_SEED = 707
CONFIRM_SEEDS = (823, 1008, 1019)
CASES = {
    (32, 1024, 5120): "dsl_32x64_split16",
    (4096, 4096, 4096): "dsl_wide_k32",
}
CUTLASS_OPTIONS = (
    # The shim computes C^T, so its tile dimensions are ordered (N, M, K).
    ("cutlass_default", (256, 128, 16), 1),
    ("cutlass_64", (64, 64, 16), 1),
    ("cutlass_128", (128, 128, 16), 1),
    ("cutlass_split4", (64, 64, 16), 4),
    ("cutlass_split8", (64, 64, 16), 8),
)
TRITON_OPTIONS = ("triton_default", "triton_small", "triton_wide", "triton_large")
METHOD_ERRORS = (ImportError, OSError, RuntimeError, ValueError, subprocess.CalledProcessError)


def shape_key(shape):
    return "x".join(map(str, shape))


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("expected a nonnegative integer")
    return number


def positive(value):
    number = nonnegative(value)
    if number == 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def gpu_name(device):
    return bytes(checked(cuda.cuDeviceGetName(128, device))).split(b"\0", 1)[0].decode()


def candidate_groups(exp, backends, cute_method):
    for backend in backends:
        if backend == "cute":
            yield ("cute",), [(cute_method, "cute", None)], None
        elif backend == "triton":
            yield ("triton",), [(name, "triton", None) for name in TRITON_OPTIONS], None
        else:
            roles = ("cublaslt_default", "cublaslt_tuned") if backend == "cublaslt" else (backend,)
            if exp.backend is None:
                yield roles, [], exp.backend_error
            elif backend == "cublas":
                yield roles, [("cublas", "cublas", None)], None
            elif backend == "cublaslt":
                catalog = exp.backend.lt_catalog()
                save(exp.out / "cublaslt-heuristics.json", catalog)
                specs = [(f"lt_{row['index']}", "lt", row) for row in catalog["algorithms"]]
                reason = f"no cuBLASLt heuristic candidates under the alignment and workspace constraints ({catalog['reason']})"
                yield roles, specs, reason
            else:
                specs = [
                    (name, "cutlass", dict(tile_id=tile[0], tile_nmk=tile, split=split))
                    for name, tile, split in CUTLASS_OPTIONS
                ]
                yield roles, specs, None


def screen(exp, backends, cute_method):
    """Rank baseline candidates on one seed and keep the requested CuTe configuration fixed."""
    panel, rows = {}, {}
    exp.inputs(SCREEN_SEED)
    for roles, specs, empty_reason in candidate_groups(exp, backends, cute_method):
        scores, rejections = {}, {}
        for spec in random.Random(SCREEN_SEED).sample(specs, len(specs)):
            name, backend, metadata = spec
            try:
                if backend == "lt" and metadata["status"]:
                    exp.backend.check_blas(
                        metadata["status"], f"cuBLASLt heuristic index {metadata['index']}"
                    )
                with ExitStack() as life:
                    method = exp.prepare(name, backend, life, metadata)
                    valid = exp.validate_method(method)
                    if not valid:
                        rejections[name] = dict(
                            status="failed", correctness="fail", reason="screening numerical check failed"
                        )
                        continue
                    exp.capture(method, life)
                    samples = []
                    for _ in range(2):
                        exp.warm(method)
                        samples.append(exp.event(method)["us"])
                    scores[name] = median(samples)
                    log(
                        exp.out / "screen.jsonl",
                        dict(
                            candidate=name,
                            backend=backend,
                            samples_us=samples,
                            median_us=scores[name],
                            config=method.metadata,
                        ),
                    )
            except METHOD_ERRORS as error:
                correctness = "not_run"
                if str(error) == "graph replay failed final-ring numerical contract":
                    correctness = "fail"
                rejections[name] = dict(
                    status="failed" if correctness == "fail" else "unavailable",
                    correctness=correctness,
                    reason=str(error),
                )
            finally:
                if name in rejections:
                    log(exp.out / "rejected.jsonl", dict(candidate=name, **rejections[name]))

        winner = min(scores, key=scores.get) if scores else None
        choices = (specs[0][0] if specs else None, winner) if len(roles) == 2 else (winner,)
        for role, chosen in zip(roles, choices):
            if chosen in scores:
                panel[role] = next(spec for spec in specs if spec[0] == chosen)
            elif chosen in rejections:
                rows[role] = dict(candidate=chosen, **rejections[chosen])
            elif rejections:
                reason = "; ".join(f"{name}: {row['reason']}" for name, row in rejections.items())
                failed = any(row["correctness"] == "fail" for row in rejections.values())
                rows[role] = dict(
                    status="failed" if failed else "unavailable",
                    correctness="fail" if failed else "not_run",
                    reason=reason,
                )
            else:
                rows[role] = dict(status="unavailable", correctness="not_run", reason=empty_reason)
    return panel, rows


def confirm(exp, panel, rows, gpu, rounds):
    """Check independent inputs and pair each surviving method in the same round."""
    if not panel:
        return [], {}
    configs, measured = {}, []
    round_ids = defaultdict(list)
    for index in range(rounds):
        round_ids[CONFIRM_SEEDS[index % len(CONFIRM_SEEDS)]].append(index + 1)
    with Monitor(gpu) as monitor:
        for seed, ids in round_ids.items():
            exp.inputs(seed)
            with ExitStack() as life:
                prepared = {}
                for role, spec in panel.items():
                    if role in rows:
                        continue
                    name = spec[0]
                    if name not in prepared:
                        try:
                            with ExitStack() as method_life:
                                method = exp.prepare(name, spec[1], method_life, spec[2])
                                configs[name] = method.metadata
                                if not exp.validate_method(method):
                                    raise ValueError("confirmation numerical check failed")
                                exp.capture(method, method_life)
                                life.enter_context(method_life.pop_all())
                            prepared[name] = method
                        except METHOD_ERRORS as error:
                            prepared[name] = str(error)
                    value = prepared[name]
                    if isinstance(value, str):
                        failed = "numerical" in value
                        rows[role] = dict(
                            status="failed",
                            correctness="fail" if failed else "not_run",
                            reason=f"seed {seed}: {value}",
                            candidate=name,
                            config=configs.get(name),
                        )
                methods = [method for method in prepared.values() if not isinstance(method, str)]
                if not methods:
                    break
                shuffled = random.Random(CONFIRM_SEEDS[0]).sample(methods, len(methods))
                for round_id in ids:
                    offset = (round_id - 1) % len(shuffled)
                    order = shuffled[offset:] + shuffled[:offset]
                    if (round_id - 1) // len(shuffled) % 2:
                        order.reverse()
                    try:
                        measured.append(exp.matched(round_id, order, monitor))
                    except RuntimeError as error:
                        for role, spec in panel.items():
                            if role not in rows:
                                rows[role] = dict(
                                    status="timing_failed",
                                    correctness="pass",
                                    reason=str(error),
                                    candidate=spec[0],
                                    config=configs[spec[0]],
                                )
                        return measured, configs
    return measured, configs


def validation_summary(out):
    checks_by_method = defaultdict(list)
    graph_checks_by_method = defaultdict(list)
    for filename, groups in (
        ("validation.jsonl", checks_by_method),
        ("graph-validation.jsonl", graph_checks_by_method),
    ):
        path = out / filename
        if path.is_file():
            with path.open() as file:
                for row in map(json.loads, file):
                    groups[row["method"]].append(row)

    return {
        method: dict(
            validation_checks=len(checks),
            validation_passes=sum(check["valid"] for check in checks),
            failed_elements=sum(check["failed"] for check in checks),
            max_abs_error=max(
                (check["max_abs"] for check in checks if check["max_abs"] is not None), default=None
            ),
            max_relative_l2=max(
                (check["relative_l2"] for check in checks if check["relative_l2"] is not None),
                default=None,
            ),
            graph_checks=len(graph_checks_by_method[method]),
            graph_passes=sum(check["valid"] for check in graph_checks_by_method[method]),
        )
        for method, checks in checks_by_method.items()
    }


def run_case(shape, cache, out, device, gpu, backends, rounds, cute_method):
    out.mkdir(parents=True)
    with ExitStack() as life:
        exp = Experiment(shape, cache, out, device, life)
        panel, rows = screen(exp, backends, cute_method)
        samples, configs = confirm(exp, panel, rows, gpu, rounds)
        samples.sort(key=lambda sample: sample["round"])
        checks = validation_summary(out)
        baseline = "cublas" if "cublas" in panel and "cublas" not in rows else None
        for role, spec in panel.items():
            name = spec[0]
            measured = [
                dict(round=sample["round"], seed=sample["seed"], us=sample["methods"][name]["us"])
                for sample in samples
                if name in sample["methods"]
            ]
            if role in rows:
                rows[role].update(samples=measured, rounds=len(measured))
                continue
            values = [sample["us"] for sample in measured]
            base_name = panel[baseline][0] if baseline else None
            ratios = [
                sample["methods"][base_name]["us"] / value
                for sample, value in zip(samples, values, strict=True) if baseline
            ]
            middle = median(values)
            rows[role] = dict(
                status="ok",
                correctness="pass",
                candidate=name,
                config=configs[name],
                median_us=middle,
                mean_us=fmean(values),
                stddev_us=stdev(values) if len(values) > 1 else None,
                min_us=min(values),
                max_us=max(values),
                tflops=2 * shape[0] * shape[1] * shape[2] / middle / 1e6,
                paired_speedup=median(ratios) if ratios else None,
                ratio_baseline=baseline,
                rounds=len(values),
                samples=measured,
            )
        for result in rows.values():
            result.update(checks.get(result.get("candidate"), {}))
        row = dict(
            shape=shape,
            cache=cache,
            l2_bytes=exp.l2,
            weight_copies=exp.weight_count,
            weight_set_bytes=exp.weight_count * exp.weight_bytes,
            roles=rows,
            artifacts=f"{out.parent.name}/{out.name}",
        )
        if exp.backend is not None:
            row["library_versions"] = exp.backend.versions()
    save(out / "summary.json", row)
    return row


def file_hash(path):
    if path.is_file():
        with path.open("rb") as file:
            return hashlib.file_digest(file, "sha256").hexdigest()
    return None


def provenance(device, gpu, driver_api):
    packages = {}
    for name in ("cuda-bindings", "numpy", "nvidia-cutlass-dsl", "triton"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    toolkit = None
    try:
        toolkit = subprocess.check_output([str(cuda_toolkit() / "bin/nvcc"), "--version"], text=True).strip()
    except (OSError, RuntimeError, subprocess.CalledProcessError):
        pass

    def git(*args, path=ROOT):
        result = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    source = fingerprint() if packages["nvidia-cutlass-dsl"] is not None else None
    hashes = {
        name: file_hash(ROOT / name)
        for name in (
            "gemm",
            "bench/benchmark.py",
            "bench/harness.py",
            "bench/tf32_backends.cu",
            "bench/cutlass_tf32.cuh",
            "bench/tf32_triton.py",
            "src/cute/tf32_api.py",
            "build/libtf32_backends.so",
        )
    }
    changes = git("status", "--porcelain", "--untracked-files=no")
    return dict(
        commit=git("rev-parse", "HEAD"),
        tracked_files_modified=bool(changes) if changes is not None else None,
        gpu=gpu_name(device),
        cuda_driver_api=driver_api,
        nvidia_driver=Clocks(gpu).driver_version(),
        python=platform.python_version(),
        packages=packages,
        nvcc=toolkit,
        cutlass_cpp=git("rev-parse", "HEAD", path=ROOT / "third_party/cutlass"),
        dispatch_sha256=file_hash(TABLE),
        dispatch_fingerprint=source,
        source_sha256=hashes,
    )


def selected_config(row):
    config = row.get("config") or {}
    return config.get("selected_method", row.get("candidate", "-"))


def write_results(out, data):
    save(out / "comparison.json", data)
    fields = (
        "shape", "backend", "selected_config", "rounds", "median_us", "mean_us", "stddev_us",
        "paired_speedup", "ratio_baseline", "correctness", "max_abs_error", "max_relative_l2",
        "status", "reason",
    )
    sample_fields = ("shape", "cache", "artifacts", "backend", "candidate", "round", "seed", "us")
    with (out / "comparison.csv").open("w", newline="") as summary_file, (out / "samples.csv").open(
        "w", newline=""
    ) as sample_file:
        summary_writer = csv.DictWriter(summary_file, fieldnames=fields)
        sample_writer = csv.DictWriter(sample_file, fieldnames=sample_fields)
        summary_writer.writeheader()
        sample_writer.writeheader()
        for case in data["cases"]:
            shared = {key: case[key] for key in ("cache", "artifacts")}
            shared["shape"] = shape_key(case["shape"])
            for role, row in case["roles"].items():
                result = dict(
                    shared,
                    **row,
                    backend=role,
                    selected_config=selected_config(row),
                )
                summary_writer.writerow({key: result.get(key) for key in fields})
                for sample in row.get("samples", ()):
                    recorded = result | sample
                    sample_writer.writerow({key: recorded.get(key) for key in sample_fields})


def print_case(case):
    print(f"\n{shape_key(case['shape'])} (hot)")
    print(f"{'backend':<18} {'median us':>10} {'vs cuBLAS':>10} {'max abs':>10} {'rel L2':>10} {'check':>7}")
    for role, row in case["roles"].items():
        if row["status"] != "ok":
            print(f"{role:<18} {row['status']}: {row['reason']}")
            continue
        ratio = row["paired_speedup"]
        speedup = f"{ratio:.3f}x" if ratio is not None else "-"
        print(f"{role:<18} {row['median_us']:>10.2f} {speedup:>10} {row['max_abs_error']:>10.4g} "
              f"{row['max_relative_l2']:>10.4g} {row['correctness']:>7}", flush=True)


def run_benchmark(out, rounds):
    out = out or ROOT / "runs" / f"benchmark-{time.time_ns()}"
    out.mkdir(parents=True)
    gpu = checked(cuda.cuDeviceGetPCIBusId(32, current_device())).split(b"\0", 1)[0].decode()
    driver_api = checked(cuda.cuDriverGetVersion())
    with gpu_lock(gpu):
        idle(gpu)
        with primary_context() as device:
            data = dict(
                schema=2,
                target="two-shape",
                provenance=provenance(device, gpu, driver_api),
                cases=[],
                protocol=dict(
                    min_event_ms=MIN_MS,
                    screen_seed=SCREEN_SEED,
                    confirmation_seeds=CONFIRM_SEEDS,
                    rounds_per_case=rounds,
                    requested_backends=BACKENDS,
                    timing="unprofiled CUDA Graph events, paired NVML clocks",
                ),
            )
            for shape, method in CASES.items():
                dest = out / shape_key(shape) / "hot"
                row = run_case(shape, "hot", dest, device, gpu, BACKENDS, rounds, method)
                data["cases"].append(row)
                print_case(row)
                write_results(out, data)
    print(f"Summary and per-round samples written to {out}", flush=True)
    return int(any(row["status"] != "ok" for case in data["cases"] for row in case["roles"].values()))


def main(argv):
    parser = argparse.ArgumentParser(description="Compare TF32 GEMMs at 32x1024x5120 and 4096 cubed.")
    parser.add_argument("--rounds", type=positive, default=12, help="paired rounds per shape (default: 12)")
    parser.add_argument("--out", type=Path, help="new directory for CSV, JSON and raw samples")
    args = parser.parse_args(argv)
    try:
        return run_benchmark(args.out, args.rounds)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"benchmark: {error}\n")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
