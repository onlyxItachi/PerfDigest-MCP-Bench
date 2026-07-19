// main.cpp — the runtime leg of the build-loop study: a measurable hot loop
// with two deliberately opposite passes (run concurrently on two threads,
// which is the whole reason find_package(Threads) is in CMakeLists.txt),
// plus an anti-DCE checksum printed to stderr so no optimizer can prove any
// of the work is unobservable and fold it away.
#include "bl_core/api.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

// Cache-unfriendly pass: a fixed random permutation gathers from a pool
// much larger than L3, so the miss rate is real, not theoretical.
std::uint64_t cache_unfriendly_pass(const bl::GammaIndex& idx,
                                     const std::vector<double>& pool) {
  std::uint64_t acc = 0;
  const std::uint32_t* order = idx.data();
  const std::size_t n = idx.size();
  const std::size_t pool_size = pool.size();
  for (std::size_t i = 0; i < n; ++i) {
    double v = pool[order[i] % pool_size];
    acc += static_cast<std::uint64_t>(v * 1000.0) ^ (acc << 1);
  }
  return acc;
}

// Compute-bound pass: dense trig/sqrt arithmetic, no memory indirection.
double compute_bound_pass(const std::vector<double>& data, int repeats) {
  double total = 0.0;
  for (int r = 0; r < repeats; ++r) {
    total += bl::alpha_reduce(data.data(), data.size());
  }
  return total;
}

}  // namespace

int main() {
  constexpr std::size_t kPoolElems = 8'000'000;    // ~64 MB of doubles, > L3
  constexpr std::size_t kIndexElems = 20'000'000;  // gather count
  constexpr std::size_t kComputeElems = 200'000;
  constexpr int kComputeRepeats = 20;

  std::vector<double> pool(kPoolElems);
  for (std::size_t i = 0; i < kPoolElems; ++i) {
    pool[i] = static_cast<double>(i % 97) * 0.5;
  }

  bl::GammaIndex idx(kIndexElems);

  std::vector<double> compute_buf(kComputeElems);
  for (std::size_t i = 0; i < kComputeElems; ++i) {
    compute_buf[i] = static_cast<double>((i * 2654435761u) % 10000) * 0.001;
  }

  std::uint64_t cache_checksum = 0;
  double compute_checksum = 0.0;

  const auto t0 = Clock::now();
  std::thread cache_thread(
      [&] { cache_checksum = cache_unfriendly_pass(idx, pool); });
  std::thread compute_thread(
      [&] { compute_checksum = compute_bound_pass(compute_buf, kComputeRepeats); });
  cache_thread.join();
  compute_thread.join();
  const auto t1 = Clock::now();

  const std::string norm = bl::beta_normalize("  Build   Loop   Bench  ");
  const std::size_t tokens = bl::beta_token_count(norm);
  const std::uint64_t hash = bl::alpha_hash_mix(0xC0FFEEULL, 200'000);
  const unsigned long long heavy = bl::heavy_checksum();

  const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // Anti-DCE: fold every pass's output into one checksum printed to stderr.
  const unsigned long long final_checksum =
      cache_checksum ^ static_cast<unsigned long long>(compute_checksum * 1000.0) ^
      tokens ^ hash ^ heavy;

  std::cerr << "checksum=" << final_checksum << " tokens=" << tokens << "\n";
  std::cout << "cache_unfriendly+compute_bound (2 threads) elapsed_ms=" << elapsed_ms
            << " norm=\"" << norm << "\"\n";
  return 0;
}
