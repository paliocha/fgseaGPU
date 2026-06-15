// fgsea_qvalue.h — Storey-Tibshirani q-values plus a discrete-aware
// mid-p variant for tests like the hypergeometric.
//
// Continuous q-values:
//   pi_0  ~  proportion of true nulls, estimated from the high-p tail
//   q(p)  =  pi_0 * m * p / rank(p), monotonised by a running min from below
//
// We pick the smoothing parameter lambda by the bootstrap rule of Storey
// (2002): for each candidate lambda, estimate the MSE of pi_0(lambda)
// against the empirically minimal pi_0 across the lambda grid, and choose
// the lambda that minimises mean bootstrap MSE. This is the same rule
// implemented by Bioconductor's `qvalue::pi0est(pi0.method="bootstrap")`
// and is the recommended default for small to moderate m.
//
// Discrete q-values: we transform each p-value into a mid-p (Heyse 2011 /
// Lancaster) and then apply the continuous machinery on the mid-p
// sequence. For one-sided hypergeometric tests,
//
//   midP(q; N, m, k) = P(X >= q) - 0.5 * P(X = q)
//
// which corrects the well-known conservatism of discrete one-sided p-values
// without introducing the randomisation of Habiger-style q-values.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fgsea::qvalue {

struct StoreyOptions {
    std::vector<double> lambdaGrid;    // empty -> default 0.05..0.95 step 0.05
    std::int64_t        bootstrap{100};
    bool                pi0Bootstrap{true};   // false => min over lambda
    double              pi0Floor{0.01};
    std::uint64_t       seed{0x5EED};
};

struct Result {
    double               pi0{1.0};
    std::vector<double>  qvalues;
};

namespace detail {

inline std::vector<double> defaultLambdaGrid() {
    std::vector<double> g;
    g.reserve(19);
    for (int i = 1; i <= 19; ++i) g.push_back(0.05 * static_cast<double>(i));
    return g; // 0.05, 0.10, ..., 0.95
}

// pi_0(lambda) computed from a *sorted-ascending* p-value vector via bisect.
inline double pi0FromSorted(std::vector<double> const& sortedP, double lambda) {
    auto const m = static_cast<double>(sortedP.size());
    if (m <= 0.0 || lambda >= 1.0) return 1.0;
    auto const it = std::ranges::upper_bound(sortedP, lambda);
    auto const above = static_cast<double>(std::distance(it, sortedP.end()));
    return std::clamp(above / (m * (1.0 - lambda)), 0.0, 1.0);
}

// Bootstrap lambda selection per Storey (2002). The outer cost is dominated
// by re-sorting each bootstrap sample (O(m log m)) — bisecting against the
// lambda grid is then O(L log m) per replicate, well under the sort.
inline double pi0Bootstrap(std::vector<double> const& pvals,
                           std::vector<double> const& lambdas,
                           std::int64_t B,
                           std::uint64_t seed)
{
    auto sortedP = pvals;
    std::ranges::sort(sortedP);

    auto const L = lambdas.size();
    std::vector<double> pi0L(L);
    for (std::size_t i = 0; i < L; ++i)
        pi0L[i] = pi0FromSorted(sortedP, lambdas[i]);

    double const pi0Min = *std::ranges::min_element(pi0L);

    std::vector<double> mse(L, 0.0);
    std::mt19937_64 rng(seed);
    auto const m = pvals.size();
    std::uniform_int_distribution<std::size_t> pick(0, m - 1);
    std::vector<double> sample(m);

    for (std::int64_t b = 0; b < B; ++b) {
        for (std::size_t i = 0; i < m; ++i) sample[i] = pvals[pick(rng)];
        std::ranges::sort(sample);
        for (std::size_t i = 0; i < L; ++i) {
            double const e = pi0FromSorted(sample, lambdas[i]) - pi0Min;
            mse[i] += e * e;
        }
    }
    auto const best = static_cast<std::size_t>(
        std::distance(mse.begin(), std::ranges::min_element(mse)));
    return std::clamp(pi0L[best], 0.0, 1.0);
}

} // namespace detail

// Storey-Tibshirani q-values from a vector of p-values.
[[nodiscard]] inline Result storey(std::vector<double> const& pvals,
                                   StoreyOptions opts = {})
{
    if (pvals.empty()) return {1.0, {}};
    for (double p : pvals) {
        if (!(p >= 0.0 && p <= 1.0)) {
            throw std::invalid_argument(
                "fgsea::qvalue::storey: p-values must lie in [0, 1]");
        }
    }
    if (opts.lambdaGrid.empty()) opts.lambdaGrid = detail::defaultLambdaGrid();

    // pi_0 estimation strategy depends on m:
    //   m <  kMinForStorey   → pi0 = 1.0 (classical BH; Storey is meaningless
    //                                    and the λ-bisect/bootstrap estimators
    //                                    are wildly unstable at small m).
    //   m >= kMinForStorey   → Storey 2002 bootstrap λ-selection or min-over-λ.
    // After estimation, any non-finite or sub-pi0Floor value is treated as
    // unreliable and we revert to pi0 = 1.0 (Storey collapses to BH there) —
    // that's the documented worst-case behaviour and is what qvalue::qvalue
    // does when its smoother is degenerate.
    constexpr std::size_t kMinForStorey = 100;
    double pi0;
    if (pvals.size() < kMinForStorey) {
        pi0 = 1.0;
    } else if (opts.pi0Bootstrap) {
        pi0 = detail::pi0Bootstrap(pvals, opts.lambdaGrid,
                                   opts.bootstrap, opts.seed);
    } else {
        auto sortedP = pvals;
        std::ranges::sort(sortedP);
        double best = 1.0;
        for (double lam : opts.lambdaGrid) {
            best = std::min(best, detail::pi0FromSorted(sortedP, lam));
        }
        pi0 = best;
    }
    if (!std::isfinite(pi0) || pi0 <= opts.pi0Floor) {
        pi0 = 1.0;
    } else {
        pi0 = std::min(pi0, 1.0);
    }

    // Compute q-values: sort p ascending, then descending running minimum.
    auto const m = pvals.size();
    std::vector<std::size_t> order(m);
    std::iota(order.begin(), order.end(), 0);
    std::ranges::sort(order, [&](std::size_t a, std::size_t b) {
        return pvals[a] < pvals[b];
    });

    std::vector<double> q(m);
    double runMin = 1.0;
    for (std::size_t rank = m; rank-- > 0; ) {
        std::size_t const idx = order[rank];
        double const raw = pi0 * static_cast<double>(m) * pvals[idx]
                         / static_cast<double>(rank + 1);
        runMin  = std::min(runMin, raw);
        q[idx]  = std::min(runMin, 1.0);
    }
    return {pi0, std::move(q)};
}

// Convenience: just the q-values, with default options.
[[nodiscard]] inline std::vector<double> qvalues(std::vector<double> const& pvals) {
    return storey(pvals).qvalues;
}

// Mid-p adjustment: for each test i, midP[i] = pvals[i] - 0.5 * pmfAtObs[i].
// `pmfAtObs[i]` is the null probability mass at the observed count.
[[nodiscard]] inline std::vector<double> midP(std::vector<double> const& pvals,
                                              std::vector<double> const& pmfAtObs)
{
    if (pvals.size() != pmfAtObs.size()) {
        throw std::invalid_argument("midP: size mismatch");
    }
    std::vector<double> out(pvals.size());
    for (std::size_t i = 0; i < pvals.size(); ++i) {
        out[i] = std::clamp(pvals[i] - 0.5 * pmfAtObs[i], 0.0, 1.0);
    }
    return out;
}

} // namespace fgsea::qvalue
