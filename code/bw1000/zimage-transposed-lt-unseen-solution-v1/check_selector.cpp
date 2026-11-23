#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

// Stub the provider calls to exercise the exact rendered selector on the host.
using hipblasStatus_t = int;
constexpr int HIPBLAS_STATUS_SUCCESS = 0;
constexpr int HIPBLAS_STATUS_INTERNAL_ERROR = 1;
constexpr int HIPBLAS_OP_T = 1;
constexpr int HIPBLAS_OP_N = 0;
constexpr int HIP_R_16BF = 14;
constexpr int HIPBLAS_COMPUTE_32F = 32;
constexpr std::size_t Workspace = 32u * 1024u * 1024u;
constexpr float Alpha = 1.0f;
constexpr float Beta = 0.0f;

struct Algorithm {
    int id;
    bool supported;
    std::size_t bytes;
};
struct hipblasLtMatmulHeuristicResult_t {
    Algorithm algo;
    int state;
    std::size_t workspaceSize;
};
struct Evidence {
    int id;
    int state;
    int support;
    int runnable;
    std::size_t advertised;
    std::size_t required;
};
struct Solution {
    Algorithm algo{};
    Evidence evidence{};
};
struct Context {
    int handle{}, desc{}, rhs{}, lhs{}, output{};
    struct {
        int catalog_count{}, examined_count{}, support_checks{}, selected_rank{-1};
    } search{};
    std::array<Solution, 4> solutions{};
    int count{};
    bool enumerated{};
};

namespace hipblaslt_ext {
enum class GemmType { HIPBLASLT_GEMM };
std::vector<hipblasLtMatmulHeuristicResult_t> catalog;
std::vector<int> tested;
int status = HIPBLAS_STATUS_SUCCESS;

int getAllAlgos(int, GemmType, int op_a, int op_b, int a, int b, int c, int d, int compute,
                std::vector<hipblasLtMatmulHeuristicResult_t>& rows) {
    assert(op_a == HIPBLAS_OP_T && op_b == HIPBLAS_OP_N);
    assert(a == HIP_R_16BF && b == a && c == a && d == a && compute == HIPBLAS_COMPUTE_32F);
    rows = catalog;
    return status;
}
int getIndexFromAlgo(Algorithm& algo) { return algo.id; }
int matmulIsAlgoSupported(int, int, const float* alpha, int, int, const float* beta, int, int,
                         Algorithm& algo, std::size_t& bytes) {
    assert(*alpha == 1.0f && *beta == 0.0f);
    tested.push_back(algo.id);
    bytes = algo.bytes;
    return algo.supported ? 0 : 2;
}
}  // namespace hipblaslt_ext

#include "find_solution.inc"

int main() {
    using namespace hipblaslt_ext;
    catalog = {{{4100, true, 960}, 0, 960}, {{4021, true, 0}, 0, 0},
               {{88, false, 0}, 0, 0}, {{88, true, 0}, 0, 0}, {{89, true, Workspace + 1}, 0, 0},
               {{90, true, Workspace}, 0, 0}, {{91, true, 0}, 0, 0}};
    Context first;
    assert(find_solutions(first) == 0 && first.enumerated && first.count == 1);
    assert(first.search.catalog_count == 7 && first.search.examined_count == 6);
    assert(first.search.support_checks == 3 && first.search.selected_rank == 5);
    assert(first.solutions[0].evidence.id == 90 && first.solutions[0].evidence.required == Workspace);
    const std::vector<int> first_checks{88, 89, 90};
    assert(tested == first_checks);

    tested.clear();
    catalog.assign(64, {{4021, true, 0}, 0, 0});
    catalog.push_back({{99, true, 0}, 0, 0});
    Context bounded;
    assert(find_solutions(bounded) == 0 && bounded.enumerated && bounded.count == 0 && tested.empty());
    assert(bounded.search.catalog_count == 65 && bounded.search.examined_count == 64);
    assert(bounded.search.selected_rank == -1);

    catalog.assign(4097, {{86, true, 0}, 0, 0});
    Context oversize;
    assert(find_solutions(oversize) == HIPBLAS_STATUS_INTERNAL_ERROR && !oversize.enumerated && tested.empty());
    status = 6;
    catalog.clear();
    Context provider_failed;
    assert(find_solutions(provider_failed) == 6 && !provider_failed.enumerated);
}
