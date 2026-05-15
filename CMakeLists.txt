# Standalone CMake build for fgseaGPU — produces a static library and a
# small CLI for benchmarking. Not used by R CMD INSTALL (Makevars does that),
# but useful for local development and CI on the C++ side.

cmake_minimum_required(VERSION 3.24)
project(fgsea_gpu
    VERSION 0.1.0
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(FGSEA_WITH_TORCH "Build the LibTorch GPU backend" OFF)
option(FGSEA_BUILD_BENCH "Build the benchmark CLI"       ON)
option(FGSEA_BUILD_TESTS "Build the C++ unit tests"      ON)

add_library(fgsea_gpu INTERFACE)
target_include_directories(fgsea_gpu INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
target_compile_features(fgsea_gpu INTERFACE cxx_std_20)
target_compile_options(fgsea_gpu INTERFACE
    -Wall -Wextra -Wpedantic -O3)

# TBB is required for libstdc++'s parallel std::execution policies.
find_package(TBB QUIET)
if(TBB_FOUND)
    target_link_libraries(fgsea_gpu INTERFACE TBB::tbb)
else()
    message(WARNING "Intel TBB not found; std::execution::par_unseq may degrade to sequential.")
endif()

if(FGSEA_WITH_TORCH)
    find_package(Torch REQUIRED)
    target_compile_definitions(fgsea_gpu INTERFACE FGSEA_WITH_TORCH)
    target_link_libraries(fgsea_gpu INTERFACE ${TORCH_LIBRARIES})
endif()

if(FGSEA_BUILD_BENCH)
    add_executable(fgsea_bench bench/fgsea_bench.cpp)
    target_link_libraries(fgsea_bench PRIVATE fgsea_gpu)
endif()

if(FGSEA_BUILD_TESTS)
    enable_testing()
    add_executable(fgsea_cpu_test tests/cpp/test_core.cpp)
    target_link_libraries(fgsea_cpu_test PRIVATE fgsea_gpu)
    add_test(NAME fgsea_cpu COMMAND fgsea_cpu_test)
endif()
