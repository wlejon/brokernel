function(brokernel_add_test name)
    add_executable(${name} ${name}.cpp no_dialogs.cpp)
    target_link_libraries(${name} PRIVATE brokernel)
    if(MSVC)
        target_compile_options(${name} PRIVATE /W4 /permissive- /utf-8)
    else()
        target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME ${name} COMMAND ${name})
    # A GPU test with no Vulkan device exits 77: skipped, not passed.
    set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

brokernel_add_test(test_smoke)
brokernel_add_test(test_lexer_parser)
brokernel_add_test(test_sema_types)
brokernel_add_test(test_modules_generics)
brokernel_add_test(test_spirv_stages)
# Every stage's module goes through spirv-val when the Vulkan SDK has it.
find_program(BROKERNEL_SPIRV_VAL spirv-val HINTS "$ENV{VULKAN_SDK}/bin")
target_compile_definitions(test_spirv_stages PRIVATE
    BROKERNEL_SPIRV_VAL="$<$<BOOL:${BROKERNEL_SPIRV_VAL}>:${BROKERNEL_SPIRV_VAL}>")
brokernel_add_test(test_cpu_gpu_parity)
brokernel_add_test(test_math_parity)
brokernel_add_test(test_fastnoise2_parity)
target_link_libraries(test_fastnoise2_parity PRIVATE FastNoise)
brokernel_add_test(test_wishlist)
brokernel_add_test(test_compile_cache)
# The bro.kernel binding, in a build with BROKERNEL_ENABLE_API.
if(TARGET brokernel_api)
    brokernel_add_test(test_kernel_api)
    target_link_libraries(test_kernel_api PRIVATE brokernel_api)
    if(WIN32)
        add_custom_command(TARGET test_kernel_api POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:test_kernel_api> $<TARGET_FILE_DIR:test_kernel_api>
            COMMAND_EXPAND_LISTS)
    endif()
endif()
