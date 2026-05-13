# fsgea-gpu

[![R-CMD-check](https://github.com/paliocha/fsgea-gpu/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/paliocha/fsgea-gpu/actions/workflows/R-CMD-check.yaml)
[![C++ tests](https://github.com/paliocha/fsgea-gpu/actions/workflows/cpp.yaml/badge.svg)](https://github.com/paliocha/fsgea-gpu/actions/workflows/cpp.yaml)
[![Lint](https://github.com/paliocha/fsgea-gpu/actions/workflows/lint.yaml/badge.svg)](https://github.com/paliocha/fsgea-gpu/actions/workflows/lint.yaml)
[![Coverage](https://github.com/paliocha/fsgea-gpu/actions/workflows/coverage.yaml/badge.svg)](https://github.com/paliocha/fsgea-gpu/actions/workflows/coverage.yaml)
[![Torch nightly](https://github.com/paliocha/fsgea-gpu/actions/workflows/torch-nightly.yaml/badge.svg)](https://github.com/paliocha/fsgea-gpu/actions/workflows/torch-nightly.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

GPU-accelerated **F**ast pre-ranked **G**ene **S**et **E**nrichment **A**nalysis.
A re-implementation of [`alserglab/fgsea`](https://github.com/alserglab/fgsea)
with the permutation null and per-pathway enrichment score computed in
batched LibTorch tensor operations that run on CUDA / ROCm / Apple Metal,
with a portable C++23 `std::execution::par_unseq` CPU fallback.

The R interface mirrors `fgsea::fgseaSimple` — drop-in replacement.

## Why

The hot loop of GSEA is the permutation null: for each pathway, compute
the enrichment-score running sum for thousands of random gene-set
permutations. Per-permutation work is independent, with no data
dependencies — embarrassingly parallel. The original `fgsea` exploits a
clever per-CPU-core segment-tree algorithm to get O(n + mK log K). On a
GPU we can collapse the entire null into three fused tensor ops (scatter,
cumsum, reduce) and run permutations by the million.

## Project layout

```
fsgea-gpu/
├── DESCRIPTION                     # R package metadata
├── NAMESPACE
├── LICENSE                         # MIT, with upstream attribution
├── R/fgsea.R                       # public R API
├── src/
│   ├── fsgea_core.hpp              # algorithm definition + types
│   ├── fsgea_cpu.hpp               # std::execution CPU backend
│   ├── fsgea_gpu.hpp               # LibTorch GPU backend (opt-in)
│   ├── fsgea_dispatch.hpp          # picks device, chunks batch
│   ├── rcpp_glue.cpp               # R <-> C++ marshalling
│   ├── RcppExports.cpp
│   ├── Makevars                    # R build (POSIX)
│   └── Makevars.win                # R build (Windows)
├── tests/testthat/                 # R-side tests, incl. vs upstream fgsea
├── tests/cpp/test_core.cpp         # standalone C++ unit tests
├── bench/fsgea_bench.cpp           # CLI benchmark
└── CMakeLists.txt                  # standalone C++ build
```

## Installation

### CPU-only (no GPU)

```r
# In R, with the working directory at the repo root:
install.packages(".", repos = NULL, type = "source")
library(fsgeaGPU)
fsgeaBackendInfo()
# $torch_built FALSE   $device "cpu"   $concurrency 16
```

### With the Torch GPU backend

```bash
# Grab LibTorch from https://pytorch.org/get-started/locally/ (cxx11 ABI)
# Then point R at it:
export FSGEA_TORCH=1
export FSGEA_TORCH_HOME=$HOME/opt/libtorch
export FSGEA_TORCH_CUDA=1   # if your LibTorch is a CUDA build
R CMD INSTALL .
```

On macOS with Apple Silicon, install a LibTorch build with MPS enabled.
The package auto-detects MPS at runtime.

## Usage

Three entry points mirror the upstream API:

```r
library(fsgeaGPU)
data(examplePathways, package = "fgsea")
data(exampleRanks,    package = "fgsea")

# Multilevel (default): accurate small p-values, CPU-parallel across pathways
res <- fgsea(examplePathways, exampleRanks)

# Simple permutation null: GPU-accelerated when LibTorch is built in
res_simple <- fgseaSimple(examplePathways, exampleRanks,
                          nperm = 10000, device = "auto")

# Over-representation analysis (Fisher / hypergeometric)
foraRes <- fora(examplePathways,
                genes    = tail(names(exampleRanks), 200),
                universe = names(exampleRanks))
```

Routing rule: `fgsea(...)` delegates to `fgseaMultilevel` by default, or to
`fgseaSimple` if you supply `nperm` — matching the upstream convention.

| Entry point        | Best for                                        | Backend                       |
|--------------------|-------------------------------------------------|-------------------------------|
| `fgseaSimple`      | Many permutations, preranked stats              | GPU (LibTorch)                |
| `fgseaMultilevel`  | Accurate small p-values (down to eps)           | CPU `par_unseq`               |
| `fgseaPhenotype`   | Classical phenotype-permutation on a matrix     | CPU `par_unseq`, GPU matmul   |
| `fora`             | Set-based over-representation                   | CPU `par_unseq`               |

`fgseaPhenotype` is the Subramanian-style mode that upstream `fgsea` skips:
input is an expression matrix plus a two-class label vector. Each
permutation shuffles the labels, recomputes a per-gene rank metric
(signal-to-noise, Welch's t, or class-mean variants) via Welford-stable
accumulation, re-ranks, and walks ES on that fresh ranking. The GPU path
turns the metric pass into a `[G, S] @ [B, S]ᵀ → [G, B]` matmul and reuses
the same scatter-cumsum-reduce kernel as `fgseaSimple` with stats varying
per column.

## Why no GPU for multilevel?

Multilevel splitting builds dependent MCMC chains: each move conditions on
the previous state, so a single chain can't be vectorised onto a GPU
thread block. We instead parallelise across *pathways* with
`std::execution::par_unseq`, which scales linearly to the number of CPU
cores for typical pathway counts (5k–25k).

## Algorithm notes

- The enrichment-score formula is the literal definition from Subramanian
  et al. 2005, with the Korotkevich et al. weighting refinement. We do
  not use the upstream square-root segment-tree heuristic — its
  inherently sequential cumulative-sum step doesn't map onto a GPU. We
  instead pay O(n) per permutation and recover the constant-factor loss
  by running tens of thousands of permutations in parallel.
- Random sampling on the device uses the argsort-of-uniform-noise trick
  for batched k-subsets. Reproducibility is guaranteed by a per-call
  splitmix-derived seed.
- Multiple-testing correction is Benjamini–Hochberg over all kept
  pathways, matching upstream.

## Multiple-testing correction: Storey q-values

The `padj` column in every output now holds a Storey-Tibshirani q-value
rather than a Benjamini–Hochberg adjusted p-value. The q-value is the
FDR analogue but with the proportion of true nulls π₀ estimated from
the high-p tail of the empirical distribution rather than assumed
to be 1. This gives strictly more power than BH whenever π₀ < 1
(typical in pathway analysis with a moderate-to-strong signal), and
reduces to BH in the worst case.

For the continuous pipelines (`fgseaSimple`, `fgseaMultilevel`) we use
the bootstrap λ-selection rule of Storey (2002): for a grid of λ in
{0.05, 0.10, …, 0.95} we estimate π₀(λ) and pick the λ minimising the
bootstrap MSE against the empirical minimum.

For `fora`, the hypergeometric tail is discrete, so naive q-values
inherit the well-known conservatism of one-sided discrete tests. We
apply the Lancaster / Heyse (2011) mid-p adjustment

    midP = P(X ≥ q) − ½ · P(X = q)

before feeding the sequence into the Storey machinery. The output
exposes both `pval` (the strict upper-tail probability) and `midP`
columns, with `padj` derived from the latter.

The estimated π₀ is attached to each output as an attribute:

```r
res <- fgsea(pathways, stats)
attr(res, "pi0")         # numeric, in (0, 1]
attr(res, "padj.method") # "storey-tibshirani"
```

If you need vanilla BH for compatibility, run
`stats::p.adjust(res$pval, "BH")` on the `pval` column.

## Differences vs upstream `fgsea`

- A zero observed ES (degenerate gene set, all-zero member statistics)
  raises a `domain_error` rather than silently returning an empty leading
  edge — fail loud is friendlier than fail quiet here.
- `padj` is a Storey q-value, not BH (see section above).
- `fora` returns an additional `midP` column.
- `fgseaMultilevel` exposes a `floored` flag indicating whether the
  reported p-value hit the `eps` floor before the level threshold
  bracketed the observed ES.
- `geseca`, `collapsePathways`, plotting helpers and the gene-set
  preparation utilities are not yet ported.

## Pushing to GitHub

The repo is initialised locally. To fork to your account:

```bash
cd ~/Documents/Claude/Projects/fsgea-gpu
gh repo create paliocha/fsgea-gpu \
    --source=. --public \
    --description "GPU/Torch-backed fast preranked GSEA" \
    --push
```

## License

MIT, with attribution to the upstream `fgsea` authors. See `LICENSE` and
`inst/UPSTREAM_LICENCE`.
