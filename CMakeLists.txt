cmake_minimum_required(VERSION 3.24)
project(brokernel LANGUAGES CXX)

# Dependencies (cmake/bro_deps.cmake): an outer project's target wins, else the
# ../<name> working tree beside the top-level project, else the head of its main,
# fetched at configure.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/bro_deps.cmake)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
    add_compile_options(/MP /permissive- /utf-8)
endif()

option(BROKERNEL_BUILD_TESTS "Build the brokernel test suite" ${PROJECT_IS_TOP_LEVEL})
option(BROKERNEL_BUILD_BENCH "Build the brokernel hill-climb benchmark harness" ${PROJECT_IS_TOP_LEVEL})

# brass: the code generator kernels lower into (MIR -> SPIR-V for the GPU,
# x64 / arm64 for the CPU).
bro_dependency(brass GITHUB wlejon/brass)

if(BROKERNEL_BUILD_TESTS OR BROKERNEL_BUILD_BENCH)
    bro_dependency(FastNoise2 GITHUB wlejon/FastNoise2 TARGET FastNoise
        OPTIONS FASTNOISE2_TOOLS=OFF FASTNOISE2_TESTS=OFF FASTNOISE2_UTILITY=OFF)
endif()

add_library(brokernel STATIC
    src/brokernel.cpp
    src/diagnostics.cpp
    src/types.cpp
    src/lexer.cpp
    src/parser.cpp
    src/stdlib_modules.cpp
    src/sema.cpp
    src/sema_check.cpp
    src/sema_builtins.cpp
    src/lowering_mir.cpp
    src/lowering_expr.cpp
    src/lowering_math.cpp
    src/lowering_ops.cpp
)
add_library(brokernel::brokernel ALIAS brokernel)
target_include_directories(brokernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(brokernel PUBLIC brass)

# --- The bronze JavaScript API (bro.kernel) ---
# Off unless asked for: bro, the one consumer that wants it, forces
# BROKERNEL_ENABLE_API ON, so nothing else needs bronze checked out.
option(BROKERNEL_ENABLE_API "Build the bronze JavaScript API (bro.kernel)" OFF)
if(BROKERNEL_ENABLE_API)
    # bronze (with brass, its code generator) through cmake/bro_deps.cmake: an
    # outer project's target, else ../bronze beside the top-level project, else
    # the head of its main, fetched at configure.
    bro_dependency(bronze GITHUB wlejon/bronze TARGET bronze_runtime_shared
                   OPTIONS BRONZE_BUILD_SHARED_RUNTIME=ON BRONZE_BUILD_TESTS=OFF)
    add_subdirectory(src/api)
endif()

if(BROKERNEL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BROKERNEL_BUILD_BENCH)
    add_executable(bench_hillclimb bench/bench_hillclimb.cpp tests/no_dialogs.cpp)
    target_link_libraries(bench_hillclimb PRIVATE brokernel FastNoise)
    target_compile_definitions(bench_hillclimb PRIVATE
        BROKERNEL_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    if(MSVC)
        target_compile_options(bench_hillclimb PRIVATE /W4 /permissive- /utf-8)
    endif()
endif()
