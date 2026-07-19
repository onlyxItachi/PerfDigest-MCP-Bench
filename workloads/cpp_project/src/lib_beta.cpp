#include "bl_core/api.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace bl {

std::string beta_normalize(const std::string& in) {
  std::string out;
  out.reserve(in.size());
  for (char c : in) {
    out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  out.erase(std::unique(out.begin(), out.end(),
                         [](char a, char b) { return a == ' ' && b == ' '; }),
            out.end());
  return out;
}

std::size_t beta_token_count(const std::string& in) {
  std::istringstream iss(in);
  std::string tok;
  std::size_t n = 0;
  while (iss >> tok) {
    ++n;
  }
  return n;
}

}  // namespace bl
