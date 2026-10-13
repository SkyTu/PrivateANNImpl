cmake_minimum_required(VERSION 3.16)
project(openfhe_native_experiment LANGUAGES CXX)

find_package(OpenFHE CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(ckks_plain_multiply ckks_plain_multiply.cpp)
target_compile_features(ckks_plain_multiply PRIVATE cxx_std_17)
target_include_directories(ckks_plain_multiply SYSTEM PRIVATE
    "${OpenFHE_INCLUDE}" "${OpenFHE_INCLUDE}/third-party/include"
    "${OpenFHE_INCLUDE}/core" "${OpenFHE_INCLUDE}/pke"
    "${OpenFHE_INCLUDE}/binfhe")
separate_arguments(OPENFHE_BUILD_FLAGS NATIVE_COMMAND "${OpenFHE_CXX_FLAGS}")
target_compile_options(ckks_plain_multiply PRIVATE ${OPENFHE_BUILD_FLAGS})
target_compile_definitions(ckks_plain_multiply PRIVATE
    EXPERIMENT_OPENFHE_VERSION="${BASE_OPENFHE_VERSION}")
target_link_libraries(ckks_plain_multiply PRIVATE OPENFHEpke OpenMP::OpenMP_CXX)

add_executable(ckks_tfhe_interop ckks_tfhe_interop.cpp)
target_compile_features(ckks_tfhe_interop PRIVATE cxx_std_17)
target_include_directories(ckks_tfhe_interop SYSTEM PRIVATE
    "${OpenFHE_INCLUDE}" "${OpenFHE_INCLUDE}/third-party/include"
    "${OpenFHE_INCLUDE}/core" "${OpenFHE_INCLUDE}/pke"
    "${OpenFHE_INCLUDE}/binfhe")
target_compile_options(ckks_tfhe_interop PRIVATE ${OPENFHE_BUILD_FLAGS})
target_link_libraries(ckks_tfhe_interop PRIVATE OPENFHEpke OpenMP::OpenMP_CXX)

add_executable(ckks_topk_cpu ckks_topk_cpu.cpp)
target_compile_features(ckks_topk_cpu PRIVATE cxx_std_17)
target_include_directories(ckks_topk_cpu SYSTEM PRIVATE
    "${OpenFHE_INCLUDE}" "${OpenFHE_INCLUDE}/third-party/include"
    "${OpenFHE_INCLUDE}/core" "${OpenFHE_INCLUDE}/pke"
    "${OpenFHE_INCLUDE}/binfhe")
target_compile_options(ckks_topk_cpu PRIVATE ${OPENFHE_BUILD_FLAGS})
target_link_libraries(ckks_topk_cpu PRIVATE OPENFHEpke OpenMP::OpenMP_CXX)
