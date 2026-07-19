// lib_template_heavy.cpp — deliberately expensive TU for the BuildDigest
// study (see ../../../results/DEMANDING_WORKLOADS_3_CPP_BUILD_LOOP_*.md).
//
// Three independent, genuine cost sources (calibrated on this host with
// clang++-22 -O2: ~2.0-2.2s standalone, see the results doc's clang_time_trace
// section for the measured phase breakdown):
//
//   1. Naive recursive constexpr: fib(27) is ~635k+ non-memoized recursive
//      calls evaluated entirely at compile time (recursion DEPTH stays ~27,
//      call COUNT is exponential) -- the classic "why is my compile slow"
//      constexpr pattern. Needs -fconstexpr-steps raised past clang's
//      1,048,576 default (see CMakeLists.txt's per-file COMPILE_OPTIONS).
//   2. Wide STL instantiation pressure: a 64-alternative std::variant and a
//      40-way std::tuple of std::map<std::string, std::vector<Row<I>>> --
//      each map/variant instantiation pulls in real machinery (allocators,
//      tree/union internals), not just a trivial struct.
//   3. An 8-deep recursively NESTED std::tuple type (InstChain<N>): each
//      level wraps the previous type in std::tuple<prev, Wrap<int, N>>, so
//      the compiler must instantiate std::tuple's own recursive _Tuple_impl
//      machinery at increasing depth. Deeper chains (tried up to 500 while
//      calibrating this file) blow up BOTH compile time and the
//      -ftime-trace JSON size (each InstantiateClass event's "detail" arg
//      carries the full nested type name) -- 8 was chosen specifically to
//      keep the trace artifact small while the fib(27) term alone already
//      clears the >=1s target.
//
// Only heavy_checksum() crosses the TU boundary (see bl_core/api.hpp) so the
// three other TUs never see (or pay for) any of this.

#include "bl_core/api.hpp"

#include <cstddef>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace bl::heavy {

// ---- (1) naive recursive constexpr ----
constexpr unsigned long long fib(unsigned n) {
  return n < 2 ? n : fib(n - 1) + fib(n - 2);
}
constexpr unsigned long long kFibResult = fib(27);

// ---- (2a) 64-alternative std::variant ----
template <typename T, std::size_t I>
struct Wrap {
  T value{};
  static constexpr std::size_t index = I;
};

template <std::size_t... Is>
constexpr auto wide_variant_tag(std::index_sequence<Is...>)
    -> std::variant<Wrap<int, Is>...>;
using WideVariant = decltype(wide_variant_tag(std::make_index_sequence<64>{}));

template <std::size_t... Is>
constexpr unsigned long long sum_variant_indices(std::index_sequence<Is...>) {
  return (Wrap<int, Is>::index + ...);
}

// ---- (2b) 40-way tuple of map<string, vector<Row<I>>> ----
template <std::size_t I>
struct Row {
  int a = 0;
  double b = 0.0;
  std::string c{};
  std::size_t tag = I;
};

template <std::size_t I>
using RowVec = std::vector<Row<I>>;

template <std::size_t... Is>
constexpr auto many_maps_tag(std::index_sequence<Is...>)
    -> std::tuple<std::map<std::string, RowVec<Is>>...>;
using ManyMaps = decltype(many_maps_tag(std::make_index_sequence<40>{}));

// ---- (3) 8-deep recursively nested std::tuple chain ----
template <unsigned N>
struct InstChain {
  using prev = typename InstChain<N - 1>::type;
  using type = std::tuple<prev, Wrap<int, N>>;
};
template <>
struct InstChain<0> {
  using type = Wrap<int, 0>;
};
using DeepChain = InstChain<8>::type;

}  // namespace bl::heavy

namespace bl {

unsigned long long heavy_checksum() {
  bl::heavy::WideVariant v = bl::heavy::Wrap<int, 7>{42};
  bl::heavy::ManyMaps maps{};
  std::get<3>(maps)["k"].push_back(bl::heavy::Row<3>{1, 2.0, "x", 3});

  unsigned long long checksum = bl::heavy::kFibResult;
  checksum += v.index();
  checksum += std::get<3>(maps).size();
  checksum += bl::heavy::sum_variant_indices(std::make_index_sequence<64>{});
  checksum += sizeof(bl::heavy::DeepChain);
  return checksum;
}

}  // namespace bl
