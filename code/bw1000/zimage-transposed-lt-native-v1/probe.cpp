#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>

namespace {

constexpr std::uint64_t M = 288;
constexpr std::uint64_t N = 3840;
constexpr std::uint64_t K = 3840;
constexpr std::size_t Workspace = 32u * 1024u * 1024u;
constexpr int MaxSolutions = 4;
constexpr float Alpha = 1.0f;
constexpr float Beta = 0.0f;

struct Evidence {
    std::int32_t solution;
    std::int32_t heuristic;
    std::int32_t support;
    std::int32_t runnable;
    std::uint64_t heuristic_workspace;
    std::uint64_t supported_workspace;
};
static_assert(std::is_standard_layout_v<Evidence> && sizeof(Evidence) == 32);

struct Solution {
    hipblasLtMatmulAlgo_t algo{};
    Evidence evidence{};
};

struct Context {
    hipblasLtHandle_t handle{};
    hipblasLtMatmulDesc_t desc{};
    hipblasLtMatmulPreference_t preference{};
    hipblasLtMatrixLayout_t rhs{};
    hipblasLtMatrixLayout_t lhs{};
    hipblasLtMatrixLayout_t output{};
    std::array<Solution, MaxSolutions> solutions{};
    int count = 0;
    bool enumerated = false;

    hipblasStatus_t close() {
        hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;
        auto retain_failure = [&status](hipblasStatus_t result) {
            if (status == HIPBLAS_STATUS_SUCCESS) {
                status = result;
            }
        };
        if (output) retain_failure(hipblasLtMatrixLayoutDestroy(output));
        if (lhs) retain_failure(hipblasLtMatrixLayoutDestroy(lhs));
        if (rhs) retain_failure(hipblasLtMatrixLayoutDestroy(rhs));
        if (preference) retain_failure(hipblasLtMatmulPreferenceDestroy(preference));
        if (desc) retain_failure(hipblasLtMatmulDescDestroy(desc));
        if (handle) retain_failure(hipblasLtDestroy(handle));
        return status;
    }
};

template <typename T>
hipblasStatus_t matmul_attribute(Context& c, hipblasLtMatmulDescAttributes_t attr, T value) {
    return hipblasLtMatmulDescSetAttribute(c.desc, attr, &value, sizeof(value));
}

hipblasStatus_t configure(Context& c) {
    hipblasStatus_t status = hipblasLtCreate(&c.handle);
    if (status != HIPBLAS_STATUS_SUCCESS) return status;
    status = hipblasLtMatmulDescCreate(&c.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F);
    if (status != HIPBLAS_STATUS_SUCCESS) return status;

    // Compute C^T = B^T A^T using column-major descriptors.
    // B is the original [K,N] transposed RHS view with stride [1,K].
    // A is the original [M,K] contiguous LHS, also a column-major [K,M] view.
    status = matmul_attribute(c, HIPBLASLT_MATMUL_DESC_TRANSA, HIPBLAS_OP_T);
    if (status != HIPBLAS_STATUS_SUCCESS) return status;
    status = matmul_attribute(c, HIPBLASLT_MATMUL_DESC_TRANSB, HIPBLAS_OP_N);
    if (status != HIPBLAS_STATUS_SUCCESS) return status;
    status = hipblasLtMatrixLayoutCreate(&c.rhs, HIP_R_16BF, K, N, K);
    if (status != HIPBLAS_STATUS_SUCCESS) return status;
    status = hipblasLtMatrixLayoutCreate(&c.lhs, HIP_R_16BF, K, M, K);
    if (status != HIPBLAS_STATUS_SUCCESS) return status;
    status = hipblasLtMatrixLayoutCreate(&c.output, HIP_R_16BF, N, M, N);
    if (status != HIPBLAS_STATUS_SUCCESS) return status;
    const auto column_order = HIPBLASLT_ORDER_COL;
    for (auto layout : {c.rhs, c.lhs, c.output}) {
        status = hipblasLtMatrixLayoutSetAttribute(layout, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                                  &column_order, sizeof(column_order));
        if (status != HIPBLAS_STATUS_SUCCESS) return status;
    }
    status = hipblasLtMatmulPreferenceCreate(&c.preference);
    if (status != HIPBLAS_STATUS_SUCCESS) return status;
    return hipblasLtMatmulPreferenceSetAttribute(c.preference,
        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &Workspace, sizeof(Workspace));
}

hipblasStatus_t find_solutions(Context& c) {
    std::array<hipblasLtMatmulHeuristicResult_t, MaxSolutions> rows{};
    auto status = hipblasLtMatmulAlgoGetHeuristic(c.handle, c.desc, c.rhs, c.lhs, c.output,
                                                  c.output, c.preference, MaxSolutions, rows.data(), &c.count);
    if (status != HIPBLAS_STATUS_SUCCESS) return status;
    if (c.count < 0 || c.count > MaxSolutions) return HIPBLAS_STATUS_INTERNAL_ERROR;
    for (int i = 0; i < c.count; ++i) {
        auto& row = rows.at(i);
        auto& target = c.solutions.at(i);
        auto& evidence = target.evidence;
        target.algo = row.algo;
        evidence.heuristic = row.state;
        evidence.heuristic_workspace = row.workspaceSize;
        evidence.solution = -1;
        evidence.support = HIPBLAS_STATUS_NOT_SUPPORTED;
        if (row.state != HIPBLAS_STATUS_SUCCESS) continue;
        const int solution = hipblaslt_ext::getIndexFromAlgo(target.algo);
        std::size_t required = 0;
        evidence.support = hipblaslt_ext::matmulIsAlgoSupported(c.handle, c.desc, &Alpha, c.rhs,
                           c.lhs, &Beta, c.output, c.output, target.algo, required);
        evidence.solution = solution;
        evidence.supported_workspace = required;
        evidence.runnable = solution >= 0 && evidence.support == HIPBLAS_STATUS_SUCCESS
                            && row.workspaceSize <= Workspace && required <= Workspace;
    }
    c.enumerated = true;
    return HIPBLAS_STATUS_SUCCESS;
}

}  // namespace

#define PROBE_API extern "C" __attribute__((visibility("default")))

PROBE_API int zimage_lt_create(void** address) {
    if (!address) return HIPBLAS_STATUS_INVALID_VALUE;
    *address = nullptr;
    auto* context = new (std::nothrow) Context();
    if (!context) return HIPBLAS_STATUS_ALLOC_FAILED;
    const auto status = configure(*context);
    if (status == HIPBLAS_STATUS_SUCCESS) {
        *address = context;
    } else {
        context->close();
        delete context;
    }
    return status;
}

PROBE_API int zimage_lt_enumerate(void* address, Evidence* out, int capacity, int* count) {
    if (!address || !out || !count || capacity < MaxSolutions) return HIPBLAS_STATUS_INVALID_VALUE;
    *count = 0;
    auto& c = *static_cast<Context*>(address);
    const auto status = c.enumerated ? HIPBLAS_STATUS_SUCCESS : find_solutions(c);
    if (status != HIPBLAS_STATUS_SUCCESS) return status;
    for (int i = 0; i < c.count; ++i) out[i] = c.solutions.at(i).evidence;
    *count = c.count;
    return status;
}

PROBE_API int zimage_lt_matmul(void* address, int ordinal, const void* lhs, const void* rhs, void* output,
                              void* workspace, std::size_t bytes, void* stream) {
    if (!address || !lhs || !rhs || !output || !workspace || bytes != Workspace)
        return HIPBLAS_STATUS_INVALID_VALUE;
    auto& c = *static_cast<Context*>(address);
    if (!c.enumerated || ordinal < 0 || ordinal >= c.count) return HIPBLAS_STATUS_INVALID_VALUE;
    const auto& selected = c.solutions.at(ordinal);
    if (!selected.evidence.runnable) return HIPBLAS_STATUS_NOT_SUPPORTED;
    return hipblasLtMatmul(c.handle, c.desc, &Alpha, rhs, c.rhs, lhs, c.lhs, &Beta, output,
                          c.output, output, c.output, &selected.algo,
                          workspace, bytes, reinterpret_cast<hipStream_t>(stream));
}

// Call only after the owner has synchronized the last submitted matmul.
PROBE_API int zimage_lt_close(void* address) {
    if (!address) return HIPBLAS_STATUS_INVALID_VALUE;
    auto* context = static_cast<Context*>(address);
    const auto status = context->close();
    delete context;
    return status;
}
