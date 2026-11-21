// Library baselines for the common FP32-buffer TF32 benchmark.
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <memory>
#include <type_traits>

#include "cutlass_tf32.cuh"

namespace {
struct Backends {
    cublasHandle_t blas = nullptr;
    cublasLtHandle_t lt = nullptr;
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr, b_desc = nullptr, c_desc = nullptr;
    cublasLtMatmulHeuristicResult_t hits[32]{};
    int count = 0;
    cublasStatus_t heuristic_status = CUBLAS_STATUS_SUCCESS;
    int m, n, k;
    cudaStream_t stream;
    void *scratch;
    size_t capacity;

    ~Backends() {
        if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
        if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
        if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
        if (desc) cublasLtMatmulDescDestroy(desc);
        if (lt) cublasLtDestroy(lt);
        if (blas) cublasDestroy(blas);
    }
};

constexpr float alpha = 1, beta = 0;
template<class Op>
int run_cutlass(Backends &p, const float *a, const float *b, float *c,
                int split, size_t *workspace, bool check) {
    Op operation;
    typename Op::Arguments args({p.n, p.m, p.k}, {b, p.n}, {a, p.k},
                               {c, p.n}, {c, p.n}, {1.f, 0.f}, split);
    *workspace = operation.get_workspace_size(args);
    if (*workspace > p.capacity) return -2;
    auto status = operation.can_implement(args);
    if (!check && status == cutlass::Status::kSuccess) {
        if constexpr (std::is_same_v<Op, cutlass_tf32::Split>)
            status = operation.initialize(args, p.scratch);
        else
            status = operation.initialize(args, p.scratch, p.stream);
        if (status == cutlass::Status::kSuccess) status = operation(p.stream);
    }
    return int(status);
}
} // namespace

extern "C" void *tf32_backends_create(int m, int n, int k, void *scratch, size_t capacity,
                                      cudaStream_t stream, int *error) {
    auto p = std::make_unique<Backends>();
    p->m = m;
    p->n = n;
    p->k = k;
    p->stream = stream;
    p->scratch = scratch;
    p->capacity = capacity;
#define TRY(call) do { *error = (call); if (*error) return nullptr; } while (0)
    TRY(cublasCreate(&p->blas));
    TRY(cublasSetStream(p->blas, stream));
    TRY(cublasSetMathMode(p->blas, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
    TRY(cublasLtCreate(&p->lt));
    TRY(cublasLtMatmulDescCreate(&p->desc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
    // The same zero-copy column-major reinterpretation as cublasGemmEx.
    TRY(cublasLtMatrixLayoutCreate(&p->a_desc, CUDA_R_32F, n, k, n));
    TRY(cublasLtMatrixLayoutCreate(&p->b_desc, CUDA_R_32F, k, m, k));
    TRY(cublasLtMatrixLayoutCreate(&p->c_desc, CUDA_R_32F, n, m, n));
    cublasLtMatmulPreference_t pref;
    TRY(cublasLtMatmulPreferenceCreate(&pref));
    uint64_t limit = capacity;
    uint32_t alignment = 16;
    auto status = cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &limit, sizeof(limit));
    for (auto attr : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                     CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES})
        if (status == CUBLAS_STATUS_SUCCESS)
            status = cublasLtMatmulPreferenceSetAttribute(pref, attr, &alignment, sizeof(alignment));
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatmulAlgoGetHeuristic(p->lt, p->desc, p->a_desc, p->b_desc,
            p->c_desc, p->c_desc, pref, 32, p->hits, &p->count);
    p->heuristic_status = status;
    cublasLtMatmulPreferenceDestroy(pref);
#undef TRY
    return p.release();
}

extern "C" void tf32_backends_destroy(void *ptr) { delete static_cast<Backends *>(ptr); }

extern "C" int tf32_cublas_version(void *ptr, int *version) {
    return cublasGetVersion(static_cast<Backends *>(ptr)->blas, version);
}

extern "C" size_t tf32_lt_version() { return cublasLtGetVersion(); }

extern "C" const char *tf32_cublas_status(int status) {
    return cublasGetStatusName(static_cast<cublasStatus_t>(status));
}

extern "C" const char *tf32_cutlass_status(int status) {
    return cutlassGetStatusString(static_cast<cutlass::Status>(status));
}

extern "C" int tf32_cublas(void *ptr, const float *a, const float *b, float *c, int pedantic) {
    auto &p = *static_cast<Backends *>(ptr);
    return cublasGemmEx(p.blas, CUBLAS_OP_N, CUBLAS_OP_N, p.n, p.m, p.k, &alpha,
        b, CUDA_R_32F, p.n, a, CUDA_R_32F, p.k, &beta, c, CUDA_R_32F, p.n,
        pedantic ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
}

extern "C" int tf32_lt_count(void *ptr, int *status) {
    auto &p = *static_cast<Backends *>(ptr);
    *status = p.heuristic_status;
    return p.count;
}

extern "C" int tf32_lt_metadata(void *ptr, int index, int *fields, size_t *workspace) {
    auto &p = *static_cast<Backends *>(ptr);
    if (index < 0 || index >= p.count) return -1;
    const auto &hit = p.hits[index];
    if (hit.state != CUBLAS_STATUS_SUCCESS) return hit.state;
    *workspace = hit.workspaceSize;
    const cublasLtMatmulAlgoConfigAttributes_t attributes[] = {
        CUBLASLT_ALGO_CONFIG_ID, CUBLASLT_ALGO_CONFIG_TILE_ID,
        CUBLASLT_ALGO_CONFIG_SPLITK_NUM, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
        CUBLASLT_ALGO_CONFIG_STAGES_ID, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
    };
    for (size_t i = 0; i < sizeof(attributes)/sizeof(attributes[0]); ++i) {
        size_t bytes = 0;
        auto status = cublasLtMatmulAlgoConfigGetAttribute(&hit.algo, attributes[i], fields+i, sizeof(int), &bytes);
        if (status != CUBLAS_STATUS_SUCCESS) fields[i] = -1;
    }
    return 0;
}

extern "C" int tf32_lt(void *ptr, int index, const float *a, const float *b, float *c) {
    auto &p = *static_cast<Backends *>(ptr);
    if (index < 0 || index >= p.count) return -1;
    const auto &hit = p.hits[index];
    if (hit.state != CUBLAS_STATUS_SUCCESS || hit.workspaceSize > p.capacity) return -2;
    return cublasLtMatmul(p.lt, p.desc, &alpha, b, p.a_desc, a, p.b_desc, &beta,
            c, p.c_desc, c, p.c_desc, &hit.algo, p.scratch, hit.workspaceSize, p.stream);
}

extern "C" int tf32_cutlass(void *ptr, int tile, int split, const float *a, const float *b,
                            float *c, size_t *workspace, int check) {
    auto &p = *static_cast<Backends *>(ptr);
    if (split > 1) return run_cutlass<cutlass_tf32::Split>(p, a, b, c, split, workspace, check);
    if (tile == 64) return run_cutlass<cutlass_tf32::Regular<64,64,16,32,32,3,1>>(
        p, a, b, c, 1, workspace, check);
    if (tile == 128) return run_cutlass<cutlass_tf32::Regular<128,128,16,64,64,4,8>>(
        p, a, b, c, 1, workspace, check);
    return run_cutlass<cutlass_tf32::Default>(p, a, b, c, 1, workspace, check);
}
