#include "bl_core/api.hpp"

#include <algorithm>
#include <numeric>
#include <random>

namespace bl {

GammaIndex::GammaIndex(std::size_t n) : perm_(n) {
  std::iota(perm_.begin(), perm_.end(), 0u);
  std::mt19937 rng(0xC0FFEEu);  // fixed seed: deterministic runtime pass
  std::shuffle(perm_.begin(), perm_.end(), rng);
}

const std::uint32_t* GammaIndex::data() const { return perm_.data(); }
std::size_t GammaIndex::size() const { return perm_.size(); }

}  // namespace bl
