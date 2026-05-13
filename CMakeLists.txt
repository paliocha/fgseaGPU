# Standalone CMake build for fsgea-gpu — produces a static library and a
# small CLI for benchmarking. Not used by R CMD INSTALL (Makevars does that),
# but useful for local development and CI on the C++ side.

cmake_minimum_required(VERSION 3.24)
project(fsgea_gpu
    VERSION 0.1.0
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(FSGEA_WITH_TORCH "Build the LibTorch GPU backend" OFF)
option(FSGEA_BUILD_BENCH "Build the benchmark CLI"       ON)
option(FSGEA_BUILD_TESTS "Build the C++ unit tests"      ON)

add_library(fsgea_gpu INTERFACE)
target_include_directories(fsgea_gpu INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
target_compile_features(fsgea_gpu INTERFACE cxx_std_23)
target_compile_options(fsgea_gpu INTERFACE
    -Wall -Wextra -Wpedantic -O3)

# TBB is required for libstdc++'s parallel std::execution policies.
find_package(TBB QUIET)
if(TBB_FOUND)
    target_link_libraries(fsgea_gpu INTERFACE TBB::tbb)
else()
    message(WARNING "Intel TBB not found; std::execution::par_unseq may degrade to sequential.")
endif()

if(FSGEA_WITH_TORCH)
    find_package(Torch REQUIRED)
    target_compile_definitions(fsgea_gpu INTERFACE FSGEA_WITH_TORCH)
    target_link_libraries(fsgea_gpu INTERFACE ${TORCH_LIBRARIES})
endif()

if(FSGEA_BUILD_BENCH)
    add_executable(fsgea_bench bench/fsgea_bench.cpp)
    target_link_libraries(fsgea_bench PRIVATE fsgea_gpu)
endif()

if(FSGEA_BUILD_TESTS)
    enable_testing()
    add_executable(fsgea_cpu_test tests/cpp/test_core.cpp)
    target_link_libraries(fsgea_cpu_test PRIVATE fsgea_gpu)
    add_test(NAME fsgea_cpu COMMAND fsgea_cpu_test)
endif()
