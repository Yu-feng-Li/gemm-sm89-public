# SM89 TF32 GEMM

CuTe DSL kernels for row-major FP32 GEMM on NVIDIA Ada Tensor Cores. The benchmark runs two shapes on an RTX 4090 with hot inputs. Times are median microseconds over 12 paired rounds, including the Split-K reduction.

| M × N × K | CuTe (µs) | cuBLAS (µs) | Paired speedup |
| --- | ---: | ---: | ---: |
| 32×1024×5120 | 9.31 | 19.06 | 2.046× |
| 4096×4096×4096 | 1,712.38 | 1,711.12 | 0.999× |

All six backend rows passed the numerical checks for both shapes.

## Run

Use Linux x86-64, an SM89 GPU, Python 3.13, a CUDA 12 toolkit with NVCC and cuBLAS, and a C++17 host compiler. The reference system uses an RTX 4090 and CUDA 12.2.

```bash
PYTHON=python3.13 bash scripts/setup.sh
bash scripts/build.sh
./gemm
```

Set `CUDA_VISIBLE_DEVICES` to select the GPU. Each backend reuses the same A and B buffers on one stream. CUDA events measure warmed Graph replay. Speedup is the median of the per-round cuBLAS time divided by the backend time. cuBLASLt reports its default and best passing candidate. CUTLASS C++ and Triton report the fastest passing preset tested.

Direct calls and Graph replay must produce finite outputs within `0.02 + 0.003 × abs(reference)` pointwise and `0.001` relative L2. The reference uses cuBLAS FP32_PEDANTIC with independent CPU FP64 spot checks. The run exits with an error if any backend fails or is unavailable. Open `comparison.csv` in the printed run directory for timings and errors, or `samples.csv` for individual rounds.

## Earlier shape tuning

An earlier RTX 4090 sweep at `3ba7f33` rechecked 54 cases. These hot-input examples each passed three paired rounds.

| M × N × K | CuTe (µs) | cuBLAS (µs) | Paired speedup |
| --- | ---: | ---: | ---: |
| 1×5120×5120 | 112.27 | 132.15 | 1.177× |
| 32×17408×5120 | 380.01 | 589.08 | 1.550× |
| 128×12288×5120 | 279.86 | 280.46 | 1.002× |
| 128×5120×17408 | 474.88 | 422.48 | 0.890× |

On that revision, Nsight Systems measured one hot 32×1024×5120 call with 9.247 µs for the partial products and 1.664 µs for reduction (15.3% of combined kernel time). For 4096³, Nsight Compute recorded 67,108,864 TF32 HMMA warp instructions for both CuTe and cuBLAS, with 254 and 226 registers per thread respectively. These captures describe the earlier kernels.

Original project code is MIT licensed. See `NOTICE` for third-party attribution.
