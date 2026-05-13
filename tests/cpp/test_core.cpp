// test_core.cpp — header-only C++ tests for the CPU path. No Torch needed.

#include "../../src/fsgea_dispatch.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

namespace {

bool near(double a, double b, double eps = 1e-9) {
    return std::abs(a - b) <= eps;
}

void test_hand_computed_es() {
    std::vector<double> stats{5, 4, 3, 2, 1, 0, -1, -2, -3, -4};
    std::vector<std::int32_t> pos{0, 3, 6}; // zero-based
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
    in.minSize = 1;
    in.maxSize = 10;
    auto out = fsgea::runFgsea(in);
    assert(out.size() == 2);
    assert(out[0].es > 0); // top set
    assert(out[1].es < 0); // bottom set
}

} // namespace

int main() {
    test_hand_computed_es();
    test_pos_clamps_at_zero();
    test_pipeline_smoke();
    std::cout << "All C++ core tests passed.\n";
    return 0;
}
