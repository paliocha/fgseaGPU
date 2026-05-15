// fgsea_bench.cpp — minimal CLI benchmark for the C++ core.
//
// Generates a synthetic ranked stat vector and N random gene sets of size k,
// then runs fgsea with the requested device. Prints wall time.
//
// Usage: fgsea_bench [n=20000] [p=500] [k=100] [nperm=1000] [device=auto]

#include "../src/fgsea_dispatch.h"

#include <chrono>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
    std::int64_t n     = argc > 1 ? std::stoll(argv[1]) : 20'000;
    std::int64_t P     = argc > 2 ? std::stoll(argv[2]) : 500;
    std::int64_t k     = argc > 3 ? std::stoll(argv[3]) : 100;
    std::int64_t nperm = argc > 4 ? std::stoll(argv[4]) : 1000;
    std::string  dev   = argc > 5 ? argv[5] : "auto";

    std::mt19937_64 rng(0xC0FFEE);
    std::normal_distribution<double> nd;

    fgsea::FgseaInput in;
    in.stats.resize(n);
    for (auto& s : in.stats) s = nd(rng);
    std::ranges::sort(in.stats, std::ranges::greater());

    in.pathwayNames.reserve(static_cast<std::size_t>(P));
    in.pathwayPositions.reserve(static_cast<std::size_t>(P));
    for (std::int64_t p = 0; p < P; ++p) {
        std::vector<std::int32_t> pos(static_cast<std::size_t>(k));
        fgsea::sampleWithoutReplacement(n, k, std::span<std::int32_t>(pos), rng);
        in.pathwayPositions.emplace_back(std::move(pos));
        in.pathwayNames.push_back("pw_" + std::to_string(p));
    }

    in.nperm      = nperm;
    in.deviceHint = dev;

    auto t0 = std::chrono::steady_clock::now();
    auto res = fgsea::runFgsea(in);
    auto t1 = std::chrono::steady_clock::now();

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "fgsea_bench: n=" << n << " P=" << P << " k=" << k
              << " nperm=" << nperm << " device=" << dev
              << "  pathways_kept=" << res.size()
              << "  wall=" << ms << " ms\n";
    return 0;
}
