// bl_core/api.hpp — the public surface of the bl_core static library.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace bl {

// lib_alpha.cpp — numeric reduction pass (feeds main's compute-bound loop).
double alpha_reduce(const double* data, std::size_t n);
std::uint64_t alpha_hash_mix(std::uint64_t seed, std::size_t rounds);

// lib_beta.cpp — small string-processing pass.
std::string beta_normalize(const std::string& in);
std::size_t beta_token_count(const std::string& in);

// lib_gamma.cpp — a fixed random permutation used to drive main's
// cache-unfriendly indirection pass.
class GammaIndex {
 public:
  explicit GammaIndex(std::size_t n);
  const std::uint32_t* data() const;
  std::size_t size() const;

 private:
  std::vector<std::uint32_t> perm_;
};

// lib_template_heavy.cpp — the deliberately expensive TU. Only ONE function
// crosses the TU boundary; everything that makes the compile slow
// (the variant/tuple/map instantiation pressure, the recursive constexpr
// fib) stays file-local so it cannot leak into (or slow down) the other
// three TUs.
unsigned long long heavy_checksum();

}  // namespace bl
