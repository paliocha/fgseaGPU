// test_core.cpp — header-only C++ tests for the CPU path. No Torch needed.

#include "../../src/fsgea_dispatch.hpp"
#include "../../src/fsgea_multilevel.hpp"
#include "../../src/fsgea_fora.hpp"

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
    test_fora_matches_hypergeometric();
    std::cout << "All C++ core tests passed.\n";
    return 0;
}
