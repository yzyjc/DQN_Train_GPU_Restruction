cmake_minimum_required(VERSION 3.20)
project(ROCm_Drive_v2 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ROCm installation path
set(ROCM_PATH "D:/AMD_ROCm" CACHE PATH "ROCm installation path")
set(HIPCC "${ROCM_PATH}/bin/hipcc.exe")

# Source files (C++ only, HIP compiled separately)
set(CPP_SOURCES
    dllmain_v2.cpp
    src/rocm_accelerator_v2.cpp
)

# Create DLL
add_library(rocm_drive SHARED ${CPP_SOURCES})

# Include directories
target_include_directories(rocm_drive PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ROCM_PATH}/include
)

# Link HIP and hipBLAS libraries
target_link_libraries(rocm_drive PRIVATE
    ${ROCM_PATH}/lib/amdhip64.lib
    ${ROCM_PATH}/lib/hipblas.lib
)

# Preprocessor definitions
target_compile_definitions(rocm_drive PRIVATE
    ROCMDRIVE_EXPORTS
    _WINDOWS
    _USRDLL
    __HIP_PLATFORM_AMD__
)

# Compile HIP kernel separately with hipcc
set(HIP_KERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_kernels.hip")
set(HIP_KERNEL_OBJ "${CMAKE_CURRENT_BINARY_DIR}/gpu_kernels.obj")

add_custom_command(
    OUTPUT ${HIP_KERNEL_OBJ}
    COMMAND ${HIPCC} --offload-arch=gfx1151 -c ${HIP_KERNEL_SRC} -o ${HIP_KERNEL_OBJ}
            -I${ROCM_PATH}/include
            -D__HIP_PLATFORM_AMD__
            --std=c++20
    DEPENDS ${HIP_KERNEL_SRC}
    COMMENT "Compiling HIP kernels with hipcc (gfx1151)"
)

add_custom_target(hip_kernels DEPENDS ${HIP_KERNEL_OBJ})

# Link HIP object file into DLL
target_sources(rocm_drive PRIVATE ${HIP_KERNEL_OBJ})
add_dependencies(rocm_drive hip_kernels)

# Set output directory
set_target_properties(rocm_drive PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
