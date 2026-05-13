// test_core.cpp — header-only C++ tests for the CPU path. No Torch needed.

#include "../../src/fsgea_dispatch.h"
#include "../../src/fsgea_multilevel.h"
#include "../../src/fsgea_fora.h"
#include "../../src/fsgea_phenotype.h"
#include "../../src/fsgea_qvalue.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

bool near(double a, double b, double eps = 1e-9) {
    return std::abs(a - b) <= eps;
}

void test_hand_computed_es() {
    std::vector<double> stats{5, 4, 3, 2, 1, 0, -1, -2, -3, -4};
    std::vector<std::int32_t> pos{0, 3, 6};
    auto r = fsgea::calcEs(std::span<double const>(stats),
                           std::span<std::int32_t const>(pos),
                           1.0, fsgea::ScoreType::Std);
    assert(near(r.es, 0.625));
}

void test_pos_clamps_at_zero() {
    std::vector<double> stats{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    std::vector<std::int32_t> pos{7, 8, 9};
    auto r = fsgea::calcEs(std::span<double const>(stats),
                           std::span<std::int32_t const>(pos),
                           1.0, fsgea::ScoreType::Pos);
    assert(r.es >= 0.0);
}

void test_pipeline_smoke() {
    fsgea::FgseaInput in;
    in.stats = {3, 2, 1, 0, -1, -2, -3};
    in.pathwayNames = {"top3", "bot3"};
    in.pathwayPositions = {{0, 1, 2}, {4, 5, 6}};
    in.nperm = 50;
    auto out = fsgea::runFgsea(in);
    assert(out.size() == 2);
    assert(out[0].es > 0);
    assert(out[1].es < 0);
}

void test_zero_es_throws() {
    std::vector<double> stats(10, 0.0);
    std::vector<std::int32_t> pos{0, 1, 2};
    bool threw = false;
    try {
        fsgea::cpu::observedEs(std::span<double const>(stats),
                               std::span<std::int32_t const>(pos),
                               1.0, fsgea::ScoreType::Std);
    } catch (std::domain_error const&) { threw = true; }
    assert(threw);
}

void test_multilevel_basic() {
    std::vector<double> stats(200);
    for (std::size_t i = 0; i < stats.size(); ++i)
        stats[i] = static_cast<double>(200 - i);

    std::vector<std::vector<std::int32_t>> pw = {
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},     // top 10 — strong enrichment
    };
    fsgea::multilevel::Config cfg;
    cfg.sampleSize = 11;
    cfg.eps = 1e-8;
    cfg.seed = 1;

    auto res = fsgea::multilevel::runMultilevel(
        std::span<double const>(stats), pw, 1.0,
        fsgea::ScoreType::Pos, cfg);
    assert(res.size() == 1);
    assert(res[0].es > 0);
    assert(res[0].pval > 0 && res[0].pval < 1);
    assert(res[0].log2err >= 0);
}

void test_qvalue_monotone_and_bounded_by_bh() {
    // Mixture: 80 nulls (uniform) + 20 strong signals (small p).
    std::vector<double> ps;
    ps.reserve(100);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> ud;
    for (int i = 0; i < 80; ++i) ps.push_back(ud(rng));
    for (int i = 0; i <  20; ++i) ps.push_back(ud(rng) * 0.005);

    auto r = fsgea::qvalue::storey(ps);
    assert(r.qvalues.size() == ps.size());
    // pi_0 in (0, 1]
    assert(r.pi0 > 0.0 && r.pi0 <= 1.0);

    // Sort by p and verify monotone q.
    std::vector<std::size_t> order(ps.size());
    std::iota(order.begin(), order.end(), 0);
    std::ranges::sort(order, [&](std::size_t a, std::size_t b) {
        return ps[a] < ps[b];
    });
    double prev = 0.0;
    for (auto idx : order) {
        assert(r.qvalues[idx] >= prev - 1e-12);
        assert(r.qvalues[idx] >= 0.0 && r.qvalues[idx] <= 1.0);
        prev = r.qvalues[idx];
    }

    // Hand-computed BH bound: q <= BH for any pi_0 <= 1.
    auto const m = static_cast<double>(ps.size());
    std::vector<double> bh(ps.size());
    {
        double runMin = 1.0;
        for (std::size_t rank = ps.size(); rank-- > 0; ) {
            auto idx = order[rank];
            double raw = ps[idx] * m / static_cast<double>(rank + 1);
            runMin = std::min(runMin, raw);
            bh[idx] = std::min(runMin, 1.0);
        }
    }
    for (std::size_t i = 0; i < ps.size(); ++i) {
        assert(r.qvalues[i] <= bh[i] + 1e-9);
    }
}

void test_midp_bounded_by_p() {
    // For a 1-sided discrete test, midP <= P always.
    std::vector<double> p = {0.10, 0.05, 0.01, 0.30};
    std::vector<double> pmf = {0.04, 0.02, 0.005, 0.10};
    auto mp = fsgea::qvalue::midP(p, pmf);
    for (std::size_t i = 0; i < p.size(); ++i) {
        assert(mp[i] >= 0.0 && mp[i] <= p[i] + 1e-12);
    }
}

void test_edge_singleton_set() {
    // k = 1: the running sum spikes to w_0 / NR = 1 at the member's position
    // and decays to zero. ES under Std/Pos is exactly 1.
    std::vector<double> stats{3, 2, 1, -1, -2, -3};
    std::vector<std::int32_t> pos{0};
    auto r = fsgea::calcEs(std::span<double const>(stats),
                           std::span<std::int32_t const>(pos),
                           1.0, fsgea::ScoreType::Std);
    assert(near(r.es, 1.0));
    assert(r.leadingEdgeEnd.has_value());
}

void test_edge_nperm_zero_throws() {
    fsgea::FgseaInput in;
    in.stats = {3, 2, 1, -1, -2, -3};
    in.pathwayNames = {"top"};
    in.pathwayPositions = {{0, 1}};
    in.nperm = 0;
    bool threw = false;
    try { (void) fsgea::runFgsea(in); }
    catch (std::invalid_argument const&) { threw = true; }
    assert(threw);
}

void test_edge_empty_pathways() {
    fsgea::FgseaInput in;
    in.stats = {3, 2, 1, -1, -2, -3};
    in.nperm = 10;
    auto out = fsgea::runFgsea(in);
    assert(out.empty());
}

void test_edge_oversized_pathway_filtered() {
    fsgea::FgseaInput in;
    in.stats = {3, 2, 1, -1, -2, -3};
    in.pathwayNames = {"all", "ok"};
    in.pathwayPositions = {{0, 1, 2, 3, 4, 5}, {0, 1}};
    in.nperm = 50;
    in.maxSize = 4;
    auto out = fsgea::runFgsea(in);
    assert(out.size() == 1);
    assert(out[0].pathway == "ok");
}

void test_phenotype_recovers_obvious_signal() {
    // 100 genes × 10 samples (5 in each class). Top 20 genes are upregulated
    // in class A. The "top15" pathway should be strongly positive.
    constexpr std::int64_t G = 100, S = 10;
    fsgea::phenotype::Input in;
    in.n_genes = G; in.n_samples = S;
    in.exprs.resize(static_cast<std::size_t>(G * S));
    in.labels = {0,0,0,0,0, 1,1,1,1,1};

    std::mt19937_64 rng(7);
    std::normal_distribution<double> n01(0.0, 1.0);
    for (std::int64_t g = 0; g < G; ++g) {
        for (std::int64_t s = 0; s < S; ++s) {
            double v = n01(rng);
            if (g < 20 && in.labels[static_cast<std::size_t>(s)] == 0) v += 3.0;
            in.exprs[static_cast<std::size_t>(g * S + s)] = v;
        }
    }
    in.pathway_names = {"top15"};
    std::vector<std::int32_t> pw;
    for (std::int32_t g = 0; g < 15; ++g) pw.push_back(g);
    in.pathway_genes = {pw};
    in.nperm  = 200;
    in.metric = fsgea::phenotype::Metric::SignalToNoise;

    auto res = fsgea::phenotype::runPhenotype(in);
    assert(res.size() == 1);
    assert(res[0].es > 0);
    assert(res[0].pval > 0 && res[0].pval <= 1);
}

void test_phenotype_exact_via_gosper() {
    // Tiny experiment: 12 samples (6 per class). C(12, 6) = 924 — well
    // below the exact-mode threshold of 50000, so Gosper's hack should
    // enumerate every distinct labeling and produce a deterministic
    // (Monte-Carlo-free) p-value. Re-running with a different seed must
    // give bit-identical results.
    constexpr std::int64_t G = 80, S = 12;
    fsgea::phenotype::Input in;
    in.n_genes = G; in.n_samples = S;
    in.exprs.resize(static_cast<std::size_t>(G * S));
    in.labels = {0,0,0,0,0,0, 1,1,1,1,1,1};

    std::mt19937_64 rng(11);
    std::normal_distribution<double> n01(0.0, 1.0);
    for (std::int64_t g = 0; g < G; ++g) {
        for (std::int64_t s = 0; s < S; ++s) {
            double v = n01(rng);
            if (g < 15 && in.labels[static_cast<std::size_t>(s)] == 0) v += 3.0;
            in.exprs[static_cast<std::size_t>(g * S + s)] = v;
        }
    }
    in.pathway_names = {"top10"};
    std::vector<std::int32_t> pw;
    for (std::int32_t g = 0; g < 10; ++g) pw.push_back(g);
    in.pathway_genes = {pw};
    in.metric = fsgea::phenotype::Metric::SignalToNoise;

    // nperm should be ignored when exact mode kicks in.
    in.nperm = 50;
    in.seed  = 1;
    auto const r1 = fsgea::phenotype::runPhenotype(in);
    in.seed  = 9999;
    auto const r2 = fsgea::phenotype::runPhenotype(in);

    assert(r1.size() == 1 && r2.size() == 1);
    assert(r1[0].es == r2[0].es);
    assert(r1[0].pval == r2[0].pval);
    // p-value should be expressible as an exact fraction with denominator
    // 924 (= C(12, 6)), so 924 * pval is an integer.
    double const npval = 924.0 * r1[0].pval;
    assert(std::abs(npval - std::round(npval)) < 1e-9);
}

void test_gosper_enumerates_all_subsets() {
    // Walk every 3-subset of [0, 8) and check we get C(8, 3) = 56 of them.
    std::uint64_t bits = (1ULL << 3) - 1;
    std::uint64_t const limit = ((1ULL << 3) - 1) << 5;
    int count = 0;
    while (true) {
        ++count;
        if (bits == limit) break;
        bits = fsgea::gosperNext(bits);
    }
    assert(count == 56);
}

void test_fora_matches_hypergeometric() {
    // N=100, m=20, k=14, q=8: matches our R test case
    fsgea::fora::Input in;
    in.universeSize = 100;
    in.querySize    = 14;
    in.queryMembers = {0,1,2,3,4,5,6,7, 49,50,51,52,53,54};
    in.pathwayMembers = { {} };
    in.pathwayMembers[0].reserve(20);
    for (std::int32_t i = 0; i < 20; ++i) in.pathwayMembers[0].push_back(i);
    in.pathwayNames = {"pw"};

    auto res = fsgea::fora::run(in);
    assert(res.size() == 1);
    assert(res[0].overlap == 8);
    // R: phyper(7, 20, 80, 14, lower.tail=FALSE) = 0.0006...
    assert(res[0].pval > 0 && res[0].pval < 1e-3);
}

} // namespace

int main() {
    test_hand_computed_es();
    test_pos_clamps_at_zero();
    test_pipeline_smoke();
    test_zero_es_throws();
    test_multilevel_basic();
    test_qvalue_monotone_and_bounded_by_bh();
    test_midp_bounded_by_p();
    test_edge_singleton_set();
    test_edge_nperm_zero_throws();
    test_edge_empty_pathways();
    test_edge_oversized_pathway_filtered();
    test_phenotype_recovers_obvious_signal();
    test_phenotype_exact_via_gosper();
    test_gosper_enumerates_all_subsets();
    test_fora_matches_hypergeometric();
    std::cout << "All C++ core tests passed.\n";
    return 0;
}
