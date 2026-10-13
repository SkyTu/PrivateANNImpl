cmake_minimum_required(VERSION 3.30.4)
project(privateimpl LANGUAGES CXX)
include(CTest)
option(PRIVATEIMPL_BUILD_GPU "Build HEonGPU entities and encrypted tests" ON)
find_package(OpenMP REQUIRED COMPONENTS CXX)

set(TFHE_RS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tfhe-rs-install" CACHE PATH
    "TFHE-rs GPU C API installation prefix")
add_library(tfhe_rs_backend STATIC
    src/he/entity/tfhe_keys.cpp
    src/he/tfhe_select.cpp src/he/tfhe_topk.cpp
    src/utils/tfhe_crypto.cpp
    src/utils/conversion/tfhe_conversion.cpp src/utils/detail/tfhe_rs_api.cpp)
target_compile_features(tfhe_rs_backend PUBLIC cxx_std_17)
target_include_directories(tfhe_rs_backend PUBLIC src)
target_compile_definitions(tfhe_rs_backend PRIVATE
    PRIVATEIMPL_TFHE_RS_LIBRARY_DEFAULT="${TFHE_RS_ROOT}/lib/libtfhe.so"
    PRIVATEIMPL_BFV_TFHE_BRIDGE_DEFAULT="${TFHE_RS_ROOT}/lib/libbfv_tfhe_bridge.so")
target_link_libraries(tfhe_rs_backend PRIVATE ${CMAKE_DL_LIBS} OpenMP::OpenMP_CXX)
set_target_properties(tfhe_rs_backend PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(PRIVATEIMPL_BUILD_GPU)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    find_package(HEonGPU CONFIG REQUIRED)

    add_library(privateimpl STATIC
        src/he/entity/client.cpp src/he/entity/server.cpp
        src/he/bfv_distance.cpp
        src/utils/bfv_crypto.cpp src/utils/convert.cpp
        src/utils/conversion/bfv_lwe.cpp src/utils/detail/bfv_secret.cpp)
    target_compile_features(privateimpl PUBLIC cxx_std_17)
    target_include_directories(privateimpl PUBLIC src)
    target_link_libraries(privateimpl PUBLIC
        HEonGPU::heongpu CUDA::cudart OpenMP::OpenMP_CXX tfhe_rs_backend gmpxx)
    set_target_properties(privateimpl PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # Count BFV key bytes under the distance benchmark's exact parameters.
    add_executable(bfv_key_size experiment/bfv/key_size.cpp)
    target_link_libraries(bfv_key_size PRIVATE privateimpl)
    set_target_properties(bfv_key_size PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)

    # Standalone benchmark: one encrypted Top-k call, without BFV setup.
    add_executable(tfhe_topk_once experiment/tfhe_rs/topk_once.cpp)
    target_link_libraries(tfhe_topk_once PRIVATE tfhe_rs_backend CUDA::cudart)

    # Small, single-trial diagnostics for bucket width and host concurrency.
    add_executable(bucket_parallel_once experiment/tfhe_rs/bucket_parallel_once.cpp)
    target_compile_definitions(bucket_parallel_once PRIVATE PRIVATEIMPL_HAS_GPU=1)
    target_link_libraries(bucket_parallel_once PRIVATE tfhe_rs_backend CUDA::cudart)
    add_executable(bucket_max_once experiment/tfhe_rs/bucket_max_once.cpp)
    target_compile_definitions(bucket_max_once PRIVATE PRIVATEIMPL_HAS_GPU=1)
    target_link_libraries(bucket_max_once PRIVATE tfhe_rs_backend CUDA::cudart)

    # Read-only launch instrumentation and a tiny CUDA concurrency probe.
    add_library(cuda_launch_probe SHARED experiment/tfhe_rs/cuda_launch_probe.cpp)
    target_compile_features(cuda_launch_probe PRIVATE cxx_std_17)
    target_link_libraries(cuda_launch_probe PRIVATE CUDA::cudart ${CMAKE_DL_LIBS})
    add_executable(cuda_overlap_probe experiment/tfhe_rs/cuda_overlap_probe.cu)
    target_compile_features(cuda_overlap_probe PRIVATE cuda_std_17)
    target_link_libraries(cuda_overlap_probe PRIVATE CUDA::cudart)
    set(PRIVATEIMPL_CUPTI_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cupti-13.0/nvidia/cu13")
    if(EXISTS "${PRIVATEIMPL_CUPTI_ROOT}/lib/libcupti.so.13")
        add_library(cupti_overlap_probe SHARED experiment/tfhe_rs/cupti_overlap_probe.cpp)
        target_compile_features(cupti_overlap_probe PRIVATE cxx_std_17)
        target_include_directories(cupti_overlap_probe PRIVATE "${PRIVATEIMPL_CUPTI_ROOT}/include")
        target_link_libraries(cupti_overlap_probe PRIVATE CUDA::cudart ${CMAKE_DL_LIBS}
            "${PRIVATEIMPL_CUPTI_ROOT}/lib/libcupti.so.13")
    endif()

    # One continuous encrypted distance -> conversion -> bucket Top-k run.
    add_executable(selection_pipeline_once experiment/tfhe_rs/selection_pipeline_once.cpp)
    target_compile_definitions(selection_pipeline_once PRIVATE PRIVATEIMPL_HAS_GPU=1)
    target_link_libraries(selection_pipeline_once PRIVATE privateimpl)
    set_target_properties(selection_pipeline_once PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)

    # Requested large components; the BFV-side oblivious gather is not implemented.
    find_package(Threads REQUIRED)
    add_executable(cluster_bucket_once experiment/tfhe_rs/cluster_bucket_once.cpp)
    target_compile_definitions(cluster_bucket_once PRIVATE PRIVATEIMPL_HAS_GPU=1)
    target_link_libraries(cluster_bucket_once PRIVATE privateimpl Threads::Threads)
    set_target_properties(cluster_bucket_once PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)

    # Diagnostic comparison of CPU-reference BFV packing/conversion paths.
    add_executable(bfv_conversion_diagnose experiment/bfv_conversion/diagnose.cpp)
    target_compile_definitions(bfv_conversion_diagnose PRIVATE PRIVATEIMPL_HAS_GPU=1)
    target_link_libraries(bfv_conversion_diagnose PRIVATE privateimpl)
    set_target_properties(bfv_conversion_diagnose PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)

endif()

# Each component has a standalone executable and is reused by the check runner.
set(COMPONENT_CHECKS bfv_distance_test bfc_tfhe_conversion tfhe_topk)
set(COMPONENT_SOURCES)
foreach(test_name IN LISTS COMPONENT_CHECKS)
    list(APPEND COMPONENT_SOURCES "test/${test_name}.cpp")
    add_executable(${test_name} "test/${test_name}.cpp")
endforeach()
add_library(component_checks OBJECT ${COMPONENT_SOURCES})
target_compile_definitions(component_checks PRIVATE PRIVATEIMPL_CHECK_NO_MAIN)
add_executable(check test/check.cpp $<TARGET_OBJECTS:component_checks>)

foreach(target IN ITEMS check component_checks ${COMPONENT_CHECKS})
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_include_directories(${target} PRIVATE src)
    target_compile_definitions(${target} PRIVATE
        PRIVATEIMPL_HAS_GPU=$<BOOL:${PRIVATEIMPL_BUILD_GPU}>)
    target_link_libraries(${target} PRIVATE tfhe_rs_backend)
    if(PRIVATEIMPL_BUILD_GPU)
        target_link_libraries(${target} PRIVATE privateimpl)
        set_target_properties(${target} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)
    endif()
endforeach()

if(EXISTS "${TFHE_RS_ROOT}/include/tfhe.h")
    target_include_directories(check PRIVATE "${TFHE_RS_ROOT}/include")
    target_compile_definitions(check PRIVATE WITH_FEATURE_GPU PRIVATEIMPL_VERIFY_TFHE_RS_ABI)
else()
    message(STATUS "TFHE-rs C API header missing; the full check will report incomplete ABI validation")
endif()

# Run each encrypted fixture once under CTest; check is the aggregate CLI entry.
add_test(NAME cpu_checks COMMAND check --cpu --report "${CMAKE_BINARY_DIR}/cpu_times.csv")
set_tests_properties(cpu_checks PROPERTIES LABELS "cpu;logic")
foreach(test_name IN LISTS COMPONENT_CHECKS)
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800)
endforeach()
set_tests_properties(bfv_distance_test tfhe_topk PROPERTIES LABELS "gpu;encrypted")
set_tests_properties(bfc_tfhe_conversion PROPERTIES LABELS "gpu;experimental;conversion")
