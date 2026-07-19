#include "bl_core/api.hpp"

#include <cmath>

namespace bl {

double alpha_reduce(const double* data, std::size_t n) {
  double acc = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    acc += std::sin(data[i]) * std::cos(data[i] * 0.5) +
           std::sqrt(std::abs(data[i]) + 1.0);
  }
  return acc;
}

std::uint64_t alpha_hash_mix(std::uint64_t seed, std::size_t rounds) {
  std::uint64_t x = seed;
  for (std::size_t i = 0; i < rounds; ++i) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x += 0x9E3779B97F4A7C15ULL;
  }
  return x;
}

}  // namespace bl
