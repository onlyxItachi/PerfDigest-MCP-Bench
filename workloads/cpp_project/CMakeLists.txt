cmake_minimum_required(VERSION 3.18)
project(cpp_build_loop_bench LANGUAGES CXX)

# Demanding-workloads pt.3 subject: a small but real C++ project observed
# through every BuildDigest lens perfdigest v1.2 has (cmake_profile,
# ninja_log, clang_time_trace, and — conditionally — linux_perf on the
# runtime binary). See ../../results/DEMANDING_WORKLOADS_3_CPP_BUILD_LOOP_*.md
# for the capture commands and the digest story.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildLoopHelpers.cmake")

find_package(Threads REQUIRED)

add_library(bl_core STATIC
  src/lib_alpha.cpp
  src/lib_beta.cpp
  src/lib_gamma.cpp
  src/lib_template_heavy.cpp
)
target_include_directories(bl_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
bl_apply_target_defaults(bl_core)

# lib_template_heavy.cpp is the deliberately expensive TU (naive recursive
# constexpr fib(27) + a 64-alternative std::variant + a 40-way tuple of
# std::map<std::string, std::vector<Row<I>>> + an 8-deep recursively nested
# std::tuple chain). clang's defaults are too low for the constexpr work, so
# raise them for this ONE source file — Clang-only flags, guarded.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set_source_files_properties(src/lib_template_heavy.cpp PROPERTIES
    COMPILE_OPTIONS "-fconstexpr-steps=33554432;-ftemplate-depth=1200"
  )
endif()

add_executable(bl_bench src/main.cpp)
target_link_libraries(bl_bench PRIVATE bl_core Threads::Threads)
bl_apply_target_defaults(bl_bench)
