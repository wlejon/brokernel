add_library(brokernel_api STATIC
    api.cpp
)

target_include_directories(brokernel_api PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
)

target_link_libraries(brokernel_api PUBLIC
    brokernel
    bronze_runtime_shared
)

target_compile_features(brokernel_api PUBLIC cxx_std_20)
