test_that("fora matches phyper on a hand-computed case", {
    universe <- paste0("g", 1:100)
    pathway  <- paste0("g", 1:20)
    query    <- paste0("g", c(1:8, 50:55)) # overlap = 8 of 14

    res <- fora(list(pw = pathway), genes = query, universe = universe)
    # phyper(q-1, m, n, k, lower.tail = FALSE) with q=8, m=20, n=80, k=14
    expected <- stats::phyper(7, 20, 80, 14, lower.tail = FALSE)
    expect_equal(res$pval, expected, tolerance = 1e-12)
    expect_equal(res$overlap, 8L)
    expect_equal(res$size,    20L)
    expect_equal(res$foldEnrichment, (8/14) / (20/100), tolerance = 1e-12)
})

test_that("fora ranks pathways by ascending p-value", {
    universe <- paste0("g", 1:200)
    pathways <- list(
        strongly_enriched = paste0("g", 1:30),
        weakly_enriched   = paste0("g", c(1:5, 100:124)),
        background        = paste0("g", 150:179)
    )
    query <- paste0("g", 1:20)
    res <- fora(pathways, query, universe)
    expect_equal(res$pathway[1], "strongly_enriched")
    expect_true(all(diff(res$pval) >= 0))
})

test_that("fora returns BH-adjusted p-values >= raw", {
    universe <- paste0("g", 1:200)
    pathways <- replicate(10, sample(universe, 20), simplify = FALSE)
    names(pathways) <- paste0("pw", seq_along(pathways))
    query <- sample(universe, 20)
    res <- fora(pathways, query, universe)
    expect_true(all(res$padj >= res$pval - 1e-12))
})

test_that("fora warns on genes outside the universe", {
    universe <- paste0("g", 1:50)
    pathways <- list(pw = paste0("g", 1:10))
    expect_warning(
        fora(pathways, c("g1", "not_in_universe"), universe),
        regexp = "belong to the universe"
    )
})
