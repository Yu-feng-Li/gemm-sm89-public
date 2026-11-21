#pragma once
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_splitk_parallel.h>

namespace cutlass_tf32 {
using namespace cutlass;
using Column = layout::ColumnMajor;

// Column-major C^T = B^T A^T, using the same row-major input buffers.
template<int M, int N, int K, int WM, int WN, int Stages, int Group>
using Regular = gemm::device::Gemm<
    float, Column, float, Column, float, Column, float,
    arch::OpClassTensorOp, arch::Sm80, gemm::GemmShape<M,N,K>,
    gemm::GemmShape<WM,WN,K>, gemm::GemmShape<16,8,8>,
    epilogue::thread::LinearCombination<float,4,float,float>,
    gemm::threadblock::GemmIdentityThreadblockSwizzle<Group>, Stages,
    4,4,false,arch::OpMultiplyAdd>;

using Default = Regular<256,128,16,64,64,3,8>;
using Split = gemm::device::GemmSplitKParallel<
    float, Column, float, Column, float, Column, float,
    arch::OpClassTensorOp, arch::Sm80, gemm::GemmShape<64,64,16>,
    gemm::GemmShape<32,32,16>, gemm::GemmShape<16,8,8>,
    epilogue::thread::LinearCombination<float,4,float,float>,
    epilogue::thread::Convert<float,4,float>, reduction::thread::ReduceAdd<float,float,4>,
    gemm::threadblock::GemmSplitKHorizontalThreadblockSwizzle,3,4,4,arch::OpMultiplyAdd>;
} // namespace cutlass_tf32
