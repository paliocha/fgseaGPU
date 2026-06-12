// fgsea_gpu.h — batched ES kernel implemented in LibTorch.
//
// Strategy: for a fixed pathway size k and batch size B, draw B random
// length-k position sets, build a [B, n] dense contribution tensor, do a
// single cumsum along dim 1, then a single amax/amin reduction. Collapses
// the permutation null into three fused tensor ops on the device.
//
// Memory bound: a single mini-batch occupies 8 * B * n bytes. The dispatcher
// (fgsea_dispatch.h) splits B into chunks under a configurable budget.
//
// Build flag: define FGSEA_WITH_TORCH and link against LibTorch to enable.

#pragma once

#include "fgsea_core.h"

#include <optional>
#include <string>
#include <string_view>

#ifdef FGSEA_WITH_TORCH
#  include <torch/torch.h>
#endif

namespace fgsea::gpu {

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
#ifdef FGSEA_WITH_TORCH
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
#ifdef FGSEA_WITH_TORCH
    return true;
#else
    return false;
#endif
}

#ifdef FGSEA_WITH_TORCH

inline torch::Device asTorchDevice(Device d) {
    switch (d) {
        case Device::CUDA: return torch::Device(torch::kCUDA);
        case Device::MPS:  return torch::Device(torch::kMPS);
        case Device::CPU:  break;
    }
    return torch::Device(torch::kCPU);
}

// MPS lacks float64 support; everywhere else we keep double precision so the
// kernels match the CPU backend bit-for-bit. The dispatcher uploads stats
// using this dtype and casts back to float64 on the CPU side.
inline torch::ScalarType computeDtype(Device d) noexcept {
    return (d == Device::MPS) ? torch::kFloat32 : torch::kFloat64;
}

// Sample B sets of k distinct positions from [0, n) on `device`. We take
// per-row top-k of uniform noise: the indices of the k largest values are a
// uniform k-subset of [0, n). topk is O(n log k) per row vs O(n log n) for
// a full argsort — a meaningful win whenever k << n. The result is then
// sorted within each row so the cumsum walk is monotonic in position.
inline torch::Tensor sampleBatchedPositions(
    std::int64_t n, std::int64_t k, std::int64_t B,
    torch::Device device, std::int64_t seed)
{
    auto gen = at::detail::createCPUGenerator(static_cast<std::uint64_t>(seed));
    auto const cpuOpts = torch::TensorOptions().dtype(torch::kFloat32);
    auto noise = torch::rand({B, n}, gen, cpuOpts).to(device, /*non_blocking=*/true);
    auto idx = std::get<1>(noise.topk(k, /*dim=*/1, /*largest=*/true, /*sorted=*/false));
    return std::get<0>(idx.sort(/*dim=*/1)).to(torch::kInt64);
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

#endif // FGSEA_WITH_TORCH

} // namespace fgsea::gpu
