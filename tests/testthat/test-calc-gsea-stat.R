test_that("calcGseaStat matches hand-computed values for tiny inputs", {
    # n = 10, set = {1, 4, 7} (1-based), gseaParam = 1
    # stats = c(5, 4, 3, 2, 1, 0, -1, -2, -3, -4)
    # |stats|^1 at set positions = c(5, 2, 1), NR = 8
    # contributions: +5/8, +2/8, +1/8 at set positions; -1/7 elsewhere
    # walk:
    #   pos 1 (set):     +5/8                          = 0.625
    #   pos 2,3 (non):   0.625 - 2/7                   = 0.339
    #   pos 4 (set):     0.339 + 2/8                   = 0.589
    #   pos 5,6 (non):   0.589 - 2/7                   = 0.304
    #   pos 7 (set):     0.304 + 1/8                   = 0.429
    #   pos 8,9,10 (non):0.429 - 3/7                   = 0.001 (~0)
    # ES = max = 0.625
    stats <- c(5, 4, 3, 2, 1, 0, -1, -2, -3, -4)
    es <- calcGseaStat(stats, c(1L, 4L, 7L), gseaParam = 1, scoreType = "std")
    expect_equal(es, 0.625, tolerance = 1e-9)
})

test_that("calcGseaStat zero-weighted reduces to Kolmogorov-Smirnov", {
    stats <- seq(10, 1)
    es <- calcGseaStat(stats, c(1L, 2L, 3L), gseaParam = 0, scoreType = "std")
    # Top-3 of 10 ranked stats: hits cluster at the top, walks +1/3 thrice
    # giving ES = 1 immediately, then decays. So ES should equal 1.
    expect_equal(es, 1.0, tolerance = 1e-9)
})

test_that("calcGseaStat returns negative ES for anti-correlated set", {
    stats <- seq(10, 1)
    es <- calcGseaStat(stats, c(8L, 9L, 10L), gseaParam = 1, scoreType = "std")
    expect_lt(es, 0)
})

test_that("scoreType=\"pos\" never returns negative ES", {
    stats <- seq(10, 1)
    es <- calcGseaStat(stats, c(8L, 9L, 10L), gseaParam = 1, scoreType = "pos")
    expect_gte(es, 0)
})
