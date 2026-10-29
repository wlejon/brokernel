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

# brass: the code generator kernels lower into (MIR -> SPIR-V for the GPU,
# x64 / arm64 for the CPU).
bro_dependency(brass GITHUB wlejon/brass)

add_library(brokernel STATIC
    src/brokernel.cpp
)
add_library(brokernel::brokernel ALIAS brokernel)
target_include_directories(brokernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(brokernel PUBLIC brass)

if(BROKERNEL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
