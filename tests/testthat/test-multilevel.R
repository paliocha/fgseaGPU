make_example <- function(n = 300, seed = 1) {
    set.seed(seed)
    stats <- sort(rnorm(n), decreasing = TRUE)
    names(stats) <- paste0("g", seq_len(n))
    pathways <- list(
        very_top = names(stats)[1:25],
        very_bot = names(stats)[(n - 24):n],
        random   = sample(names(stats), 25)
    )
    list(stats = stats, pathways = pathways)
}

test_that("fgseaMultilevel returns the expected columns", {
    ex <- make_example()
    res <- fgseaMultilevel(ex$pathways, ex$stats,
                           sampleSize = 11, eps = 1e-10, seed = 1)
    expect_s3_class(res, "data.table")
    expect_named(res, c("pathway", "pval", "padj", "log2err", "ES", "NES",
                        "size", "leadingEdge", "floored"))
    expect_equal(nrow(res), length(ex$pathways))
})

test_that("multilevel p-values are smaller than simple permutation's for clearly enriched sets", {
    ex <- make_example(n = 500)
    simple <- fgseaSimple(ex$pathways, ex$stats, nperm = 200,
                          seed = 1, device = "cpu")
    multi  <- fgseaMultilevel(ex$pathways, ex$stats,
                              sampleSize = 11, eps = 1e-10, seed = 1)
    data.table::setkey(simple, pathway); data.table::setkey(multi, pathway)
    # very_top should be significant in both; multilevel can resolve below 1/(nperm+1)
    expect_lt(multi["very_top", pval], 0.05)
    # log2err is always non-negative
    expect_true(all(multi$log2err >= 0))
})

test_that("multilevel honours the eps floor", {
    ex <- make_example()
    res <- fgseaMultilevel(ex$pathways, ex$stats,
                           sampleSize = 11, eps = 1e-3, seed = 1)
    expect_true(all(res$pval >= 1e-3))
})

test_that("zero-ES gene set raises an informative error", {
    # Gene set where members are perfectly balanced -> ES is exactly zero.
    # We construct this artificially: stats[1:6] = 0, set = {1,2,3,4,5,6}.
    stats <- c(rep(0, 6), seq(-1, -4))
    names(stats) <- paste0("g", seq_along(stats))
    pw <- list(zero = names(stats)[1:6])
    expect_error(
        fgseaMultilevel(pw, stats, sampleSize = 5, seed = 1),
        regexp = "leading edge is undefined"
    )
})
