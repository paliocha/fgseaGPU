// fsgea_gpu.hpp — batched ES kernel implemented in LibTorch.
//
// Strategy: for a fixed pathway size k and batch size B, draw B random
// length-k position sets, build a [B, n] dense contribution tensor, do a
// single cumsum along dim 1, then a single amax/amin reduction. Collapses
// the permutation null into three fused tensor ops on the device.
//
// Memory bound: a single mini-batch occupies 8 * B * n bytes. The dispatcher
// (fsgea_dispatch.hpp) splits B into chunks under a configurable budget.
//
// Build flag: define FSGEA_WITH_TORCH and link against LibTorch to enable.

#pragma once

#include "fsgea_core.hpp"

#include <optional>
#include <string>
#include <string_view>

#ifdef FSGEA_WITH_TORCH
#  include <torch/torch.h>
#endif

namespace fsgea::gpu {

enum class Device : std::uint8_t { CPU, CUDA, MPS };

inline std::string_view deviceName(Device d) noexcept {
    switch (d) {
        case Device::CPU:  return "cpu";
        case Device::CUDA: return "cuda";   // also reported for ROCm builds
        case Device::MPS:  return "mps";
    }
    return "?";
}

[[nodiscard]] inline Device resolveDevice(std::string_view hint) {
#ifdef FSGEA_WITH_TORCH
    if (hint == "cpu")  return Device::CPU;
    if (hint == "cuda" || hint == "rocm")
        return torch::cuda::is_available() ? Device::CUDA : Device::CPU;
    if (hint == "mps")
        return torch::mps::is_available()  ? Device::MPS  : Device::CPU;
    if (hint == "auto" || hint.empty()) {
        if (torch::cuda::is_available()) return Device::CUDA;
        if (torch::mps::is_available())  return Device::MPS;
        return Device::CPU;
    }
#endif
    (void)hint;
    return Device::CPU;
}

[[nodiscard]] constexpr bool torchAvailable() noexcept {
#ifdef FSGEA_WITH_TORCH
    return true;
#else
    return false;
#endif
}

#ifdef FSGEA_WITH_TORCH

inline torch::Device asTorchDevice(Device d) {
    switch (d) {
        case Device::CUDA: return torch::Device(torch::kCUDA);
        case Device::MPS:  return torch::Device(torch::kMPS);
        case Device::CPU:  break;
    }
    return torch::Device(torch::kCPU);
}

// Sample B sets of k distinct positions from [0, n) on `device`. argsort-over-
// noise is the canonical batched "random k-subset" trick: ranks of uniform
// noise are a uniform permutation, take the first k. Returns an int64 [B, k]
// tensor with each row sorted ascending so the cumsum walk is monotonic in
// position.
inline torch::Tensor sampleBatchedPositions(
    std::int64_t n, std::int64_t k, std::int64_t B,
    torch::Device device, std::int64_t seed)
{
    auto gen = at::detail::createCPUGenerator(static_cast<std::uint64_t>(seed));
    auto cpuOpts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto noise = torch::rand({B, n}, gen, cpuOpts).to(device, /*non_blocking=*/true);
    // argsort -> per-row uniform permutation; first k columns are the subset.
    auto perm   = noise.argsort(/*dim=*/1);
    auto subset = perm.slice(/*dim=*/1, 0, k);
    return std::get<0>(subset.sort(/*dim=*/1)).to(torch::kInt64);
}

// Compute B permutation ES values for a single pathway size k.
// `stats` is [n] float64 on the device.
inline torch::Tensor permEsBatchTorch(
    torch::Tensor const& stats,
    std::int64_t k,
    std::int64_t B,
    double gseaParam,
    ScoreType scoreType,
    std::int64_t seed)
{
    auto const n = stats.size(0);
    auto const opts = stats.options();
    if (k <= 0 || k >= n) return torch::zeros({B}, opts);

    auto positions = sampleBatchedPositions(n, k, B, stats.device(), seed); // [B, k]
    auto absStats  = stats.abs().pow(gseaParam);                            // [n]
    auto memberW   = absStats.index({positions});                           // [B, k]
    auto NR        = memberW.sum(/*dim=*/1, /*keepdim=*/true).clamp_min(1e-300);
    auto memberContrib = memberW / NR;                                      // [B, k]

    // step[b, j] = +w if j is a member of perm b, else -1/(n-k)
    double const negStep = 1.0 / static_cast<double>(n - k);
    auto step = torch::full({B, n}, -negStep, opts);
    step.scatter_(/*dim=*/1, positions, memberContrib);

    auto cs = step.cumsum(/*dim=*/1);
    switch (scoreType) {
        case ScoreType::Pos:
            return std::get<0>(cs.max(/*dim=*/1));
        case ScoreType::Neg:
            return std::get<0>(cs.min(/*dim=*/1));
        case ScoreType::Std: {
            auto hi = std::get<0>(cs.max(/*dim=*/1));
            auto lo = std::get<0>(cs.min(/*dim=*/1));
            return torch::where(hi.abs() >= lo.abs(), hi, lo);
        }
    }
    return cs.select(1, n - 1); // unreachable
}

// Observed ES for many pathways. Pathways have wildly varying k so we
// iterate per pathway here; the heavy lifting is the permutation null.
inline torch::Tensor observedEsBatch(
    torch::Tensor const& stats,            // [n] float64
    torch::Tensor const& packedPositions,  // [totalK] int64
    torch::Tensor const& offsets,          // [P+1] int64 (CPU)
    double gseaParam,
    ScoreType scoreType)
{
    auto const P = offsets.size(0) - 1;
    auto out = torch::empty({P}, stats.options());
    auto a = offsets.accessor<std::int64_t, 1>();
    auto absStats = stats.abs().pow(gseaParam);
    auto const n = stats.size(0);

    for (std::int64_t p = 0; p < P; ++p) {
        auto begin = a[p], end = a[p + 1];
        std::int64_t const k = end - begin;
        if (k <= 0 || k >= n) { out[p] = 0.0; continue; }

        auto pos = packedPositions.slice(0, begin, end);
        auto w   = absStats.index({pos});
        auto nr  = w.sum().clamp_min(1e-300);
        auto mem = w / nr;

        double const negStep = 1.0 / static_cast<double>(n - k);
        auto step = torch::full({n}, -negStep, stats.options());
        step.scatter_(/*dim=*/0, pos, mem);
        auto cs = step.cumsum(0);

        torch::Tensor es;
        switch (scoreType) {
            case ScoreType::Pos: es = cs.max(); break;
            case ScoreType::Neg: es = cs.min(); break;
            case ScoreType::Std: {
                auto hi = cs.max(), lo = cs.min();
                es = (hi.abs() >= lo.abs()).item<bool>() ? hi : lo;
            }
        }
        out[p] = es;
    }
    return out;
}

#endif // FSGEA_WITH_TORCH

} // namespace fsgea::gpu
