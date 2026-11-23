hipblasStatus_t find_solutions(Context& c) {
    // getAllAlgos ranks this datatype/transposition family. The supported-size
    // check below still uses the exact descriptors configured for this shape.
    std::vector<hipblasLtMatmulHeuristicResult_t> rows;
    const auto status = hipblaslt_ext::getAllAlgos(c.handle, hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
        HIPBLAS_OP_T, HIPBLAS_OP_N, HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIP_R_16BF,
        HIPBLAS_COMPUTE_32F, rows);
    if (status != HIPBLAS_STATUS_SUCCESS) return status;
    constexpr std::size_t CatalogLimit = 4096;
    constexpr std::size_t PrefixLimit = 64;
    if (rows.size() > CatalogLimit) return HIPBLAS_STATUS_INTERNAL_ERROR;
    const std::size_t limit = rows.size() < PrefixLimit ? rows.size() : PrefixLimit;
    c.search.catalog_count = static_cast<std::int32_t>(rows.size());
    std::array<int, PrefixLimit> seen{};
    std::size_t seen_count = 0;
    for (std::size_t rank = 0; rank < limit; ++rank) {
        ++c.search.examined_count;
        auto& row = rows.at(rank);
        auto algorithm = row.algo;
        const int id = hipblaslt_ext::getIndexFromAlgo(algorithm);
        if (id < 0 || id == 4100 || id == 4021) continue;
        bool duplicate = false;
        for (std::size_t previous = 0; previous < seen_count; ++previous) {
            if (seen.at(previous) == id) duplicate = true;
        }
        if (duplicate) continue;
        seen.at(seen_count++) = id;
        std::size_t required = 0;
        ++c.search.support_checks;
        const auto supported = hipblaslt_ext::matmulIsAlgoSupported(c.handle, c.desc, &Alpha, c.rhs,
            c.lhs, &Beta, c.output, c.output, algorithm, required);
        if (supported != HIPBLAS_STATUS_SUCCESS || required > Workspace) continue;
        auto& selected = c.solutions.at(0);
        selected.algo = algorithm;
        selected.evidence = Evidence{id, row.state, supported, 1, row.workspaceSize, required};
        c.count = 1;
        c.search.selected_rank = static_cast<std::int32_t>(rank);
        break;
    }
    c.enumerated = true;
    return HIPBLAS_STATUS_SUCCESS;
}
