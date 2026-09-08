cmake_minimum_required(VERSION 3.24)

project(low_precision_formats LANGUAGES CXX)

option(ENABLE_CUDA "Build CUDA software-emulation kernels when available" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(low_precision_core
    src/config.cpp
    src/file_io.cpp
    src/quantization_cpu.cpp
)
target_include_directories(low_precision_core PUBLIC include)
target_compile_options(low_precision_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)

set(LOW_PRECISION_CUDA_ENABLED OFF)
if(ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        target_sources(low_precision_core PRIVATE src/quantization_cuda.cu)
        target_link_libraries(low_precision_core PUBLIC CUDA::cudart)
        target_compile_definitions(low_precision_core PUBLIC
            LOW_PRECISION_HAS_CUDA=1
        )
        set_target_properties(low_precision_core PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_SEPARABLE_COMPILATION ON
        )
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set_property(TARGET low_precision_core PROPERTY
                CUDA_ARCHITECTURES 60 70 75 80)
        endif()
        set(LOW_PRECISION_CUDA_ENABLED ON)
    endif()
endif()

if(NOT LOW_PRECISION_CUDA_ENABLED)
    target_sources(low_precision_core PRIVATE src/quantization_cuda_stub.cpp)
endif()

add_executable(low_precision_tool src/main.cpp)
target_link_libraries(low_precision_tool PRIVATE low_precision_core)

include(CTest)
if(BUILD_TESTING)
    add_executable(low_precision_tests tests/test_quantization.cpp)
    target_link_libraries(low_precision_tests PRIVATE low_precision_core)
    target_compile_definitions(low_precision_tests PRIVATE
        LOW_PRECISION_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )
    add_test(NAME low_precision_tests COMMAND low_precision_tests)
endif()

message(STATUS "CUDA low-precision backend: ${LOW_PRECISION_CUDA_ENABLED}")
