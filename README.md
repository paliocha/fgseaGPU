# fsgea-gpu

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

```r
library(fsgeaGPU)
data(examplePathways, package = "fgsea")
data(exampleRanks,    package = "fgsea")

res <- fgsea(pathways = examplePathways,
             stats    = exampleRanks,
             nperm    = 10000,
             device   = "auto")     # picks CUDA > MPS > CPU automatically

head(res[order(pval)])
```

The output columns are identical in name, type, and ordering to
`fgsea::fgseaSimple`'s — pipelines that key off those columns work
unchanged.

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

## Differences vs upstream `fgsea`

- No `fgseaMultilevel` yet — only the simple permutation variant. The
  multilevel adaptive splitting algorithm is on the roadmap.
- No `fora` (over-representation) yet.
- No leading-edge for `scoreType = "neg"` corner cases when ES is
  exactly zero; we return an empty vector instead of erroring.

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
