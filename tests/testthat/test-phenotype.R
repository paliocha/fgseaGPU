make_phenotype_example <- function(G = 200, S = 20, n_up = 30, seed = 1) {
  set.seed(seed)
  exprs <- matrix(rnorm(G * S),
    nrow = G,
    dimnames = list(paste0("g", seq_len(G)), NULL)
  )
  labels <- rep(c("A", "B"), each = S / 2)
  # Upregulate the first n_up genes in class A.
  exprs[seq_len(n_up), labels == "A"] <-
    exprs[seq_len(n_up), labels == "A"] + 2.5
  pathways <- list(
    top = rownames(exprs)[seq_len(n_up)],
    rest = rownames(exprs)[(n_up + 1):G],
    random = sample(rownames(exprs), n_up)
  )
  list(exprs = exprs, labels = labels, pathways = pathways)
}

test_that("fgseaPhenotype recovers an obviously-enriched pathway", {
  ex <- make_phenotype_example()
  res <- fgseaPhenotype(ex$pathways, ex$exprs, ex$labels,
    nperm = 300, device = "cpu"
  )
  expect_s3_class(res, "data.table")
  expect_named(res, c(
    "pathway", "pval", "padj", "ES", "NES",
    "nMoreExtreme", "size", "leadingEdge"
  ))
  expect_equal(nrow(res), length(ex$pathways))

  data.table::setkey(res, pathway)
  expect_gt(res["top", ES], 0)
  expect_lt(res["top", pval], 0.05)
  expect_gt(res["rest", pval], res["top", pval])
})

test_that("fgseaPhenotype is reproducible across runs with the same seed", {
  ex <- make_phenotype_example()
  a <- fgseaPhenotype(ex$pathways, ex$exprs, ex$labels,
    nperm = 100, seed = 42, device = "cpu"
  )
  b <- fgseaPhenotype(ex$pathways, ex$exprs, ex$labels,
    nperm = 100, seed = 42, device = "cpu"
  )
  expect_equal(a$ES, b$ES)
  expect_equal(a$pval, b$pval)
})

test_that("fgseaPhenotype rejects malformed inputs", {
  ex <- make_phenotype_example(G = 50, S = 10, n_up = 10)
  expect_error(
    fgseaPhenotype(ex$pathways, ex$exprs, c("A", "A", "A"),
      nperm = 50, device = "cpu"
    ),
    regexp = "length"
  )
  expect_error(
    fgseaPhenotype(ex$pathways, ex$exprs,
      rep("only_one_class", ncol(ex$exprs)),
      nperm = 50, device = "cpu"
    ),
    regexp = "exactly two"
  )
})

test_that("fgseaPhenotype honours the metric argument", {
  # Use a comfortably-large signal: with G = 200 / n_up = 60 and a +2.5 shift
  # the rank-correlation between metrics is high enough that "top" wins
  # under both s2n and ttest. At smaller G / n_up Welch's t at n = 8 per
  # class is noisy enough that a noise gene's tiny in-class sample variance
  # can occasionally outrank a truly-shifted gene, leaving "rest" (which
  # by construction covers all the noise floor) as the top hit.
  ex <- make_phenotype_example(G = 200, S = 16, n_up = 60)
  res_s2n <- fgseaPhenotype(ex$pathways, ex$exprs, ex$labels,
    nperm = 100, metric = "s2n", device = "cpu"
  )
  res_t <- fgseaPhenotype(ex$pathways, ex$exprs, ex$labels,
    nperm = 100, metric = "ttest", device = "cpu"
  )
  expect_equal(attr(res_s2n, "metric"), "s2n")
  expect_equal(attr(res_t, "metric"), "ttest")
  expect_equal(res_s2n$pathway[1], "top")
  expect_equal(res_t$pathway[1], "top")
})
