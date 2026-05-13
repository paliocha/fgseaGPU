make_example <- function(n = 200, seed = 1) {
    set.seed(seed)
    stats <- sort(rnorm(n), decreasing = TRUE)
    names(stats) <- paste0("g", seq_len(n))
    pathways <- list(
        top    = names(stats)[1:20],                  # highly enriched at top
        bottom = names(stats)[(n - 19):n],            # at bottom
        random = sample(names(stats), 20),            # null
        small  = names(stats)[c(1, 5, 9)],            # tiny set
        weird  = c("not_a_gene", names(stats)[3:7])   # has NA on lookup
    )
    list(stats = stats, pathways = pathways)
}

test_that("fgsea returns the right shape", {
    ex <- make_example()
    res <- fgsea(ex$pathways, ex$stats, nperm = 200, seed = 1, device = "cpu")
    expect_s3_class(res, "data.table")
    expect_named(res, c("pathway", "pval", "padj", "ES", "NES",
                        "nMoreExtreme", "size", "leadingEdge"))
    expect_equal(nrow(res), length(ex$pathways))
})

test_that("top/bottom pathways have opposite-sign ES", {
    ex <- make_example()
    res <- fgsea(ex$pathways, ex$stats, nperm = 500, seed = 7, device = "cpu")
    setkey(res, pathway)
    expect_gt(res["top",    ES], 0)
    expect_lt(res["bottom", ES], 0)
})

test_that("p-values are in (0, 1] and adjusted >= raw on average", {
    ex <- make_example(n = 500)
    res <- fgsea(ex$pathways, ex$stats, nperm = 500, seed = 11, device = "cpu")
    expect_true(all(res$pval > 0 & res$pval <= 1))
    expect_true(all(res$padj > 0 & res$padj <= 1))
    expect_gte(mean(res$padj - res$pval), 0)
})

test_that("identical seeds give identical results", {
    ex <- make_example()
    r1 <- fgsea(ex$pathways, ex$stats, nperm = 300, seed = 42, device = "cpu")
    r2 <- fgsea(ex$pathways, ex$stats, nperm = 300, seed = 42, device = "cpu")
    expect_equal(r1$ES,   r2$ES)
    expect_equal(r1$pval, r2$pval)
})

test_that("GPU and CPU agree on observed ES (deterministic) within float tolerance", {
    skip_if_not(fsgeaBackendInfo()$torch_built, "Torch backend not built")
    skip_if(fsgeaBackendInfo()$device == "cpu", "No GPU device available")
    ex <- make_example()
    rc <- fgsea(ex$pathways, ex$stats, nperm = 100, seed = 1, device = "cpu")
    rg <- fgsea(ex$pathways, ex$stats, nperm = 100, seed = 1, device = "auto")
    setkey(rc, pathway); setkey(rg, pathway)
    expect_equal(rc$ES, rg$ES, tolerance = 1e-9)
})

test_that("fgsea matches upstream fgsea::fgseaSimple where available", {
    skip_if_not_installed("fgsea")
    ex <- make_example(n = 300)
    ours <- fgsea(ex$pathways, ex$stats, nperm = 2000, seed = 99, device = "cpu")
    theirs <- fgsea::fgseaSimple(
        pathways = ex$pathways, stats = ex$stats,
        nperm    = 2000, minSize = 1, maxSize = length(ex$stats),
        scoreType = "std", BPPARAM = BiocParallel::SerialParam())
    setkey(ours, pathway); setkey(theirs, pathway)
    # ES is deterministic given the same inputs; p-values converge with nperm.
    expect_equal(ours[theirs$pathway, ES], theirs$ES, tolerance = 1e-9)
    expect_lt(max(abs(ours[theirs$pathway, pval] - theirs$pval)), 0.05)
})
