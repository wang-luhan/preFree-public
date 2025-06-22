cmake_minimum_required(VERSION 3.17)

# Include required modules
include(CheckLanguage)

project(PreFreeProj LANGUAGES CXX C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

if(POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif()

# Set policy for CUDA module removal
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 NEW)
endif()

# CUDA settings
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES "80")

option(USE_FP64 "Enable fp64 support" ON)

if(USE_FP64)
    message(STATUS "FP64 is enabled")
    add_definitions(-Dfp64)
else()
    message(STATUS "FP64 is disabled")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB cuda_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/**/*.cu")

include_directories(${PROJECT_SOURCE_DIR}/include)

# Check if CUDA is available
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA version: ${CMAKE_CUDA_COMPILER_VERSION}")
    message(STATUS "CUDA toolkit root: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
else()
    message(FATAL_ERROR "CUDA compiler not found! Please install CUDA toolkit.")
endif()

find_package(CUDAToolkit REQUIRED COMPONENTS cusparse)

message(STATUS "CUDA include dirs: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

# Debug information
message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

# Create CUDA library
add_library(PreFreelib ${cuda_sources})
set_property(TARGET PreFreelib PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(PreFreelib PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set(CUDA_TEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/test)
set(CUDA_UTILITIES_PATH ${CMAKE_CURRENT_SOURCE_DIR}/utilities) 

file(GLOB cuda_utilities "${CUDA_UTILITIES_PATH}/*.cpp")
add_library(cuda_utilities_lib STATIC ${cuda_utilities})
target_link_libraries(cuda_utilities_lib PRIVATE CUDA::cusparse)

file(GLOB CUDA_TEST_FILES "${CUDA_TEST_PATH}/test_cdspmv.cpp")
message(STATUS "CUDA_TEST_FILES: ${CUDA_TEST_FILES}")

if(CUDA_TEST_FILES)
    add_executable(cuda_perftest ${CUDA_TEST_FILES})
    target_include_directories(cuda_perftest PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )

    target_link_libraries(cuda_perftest PRIVATE 
        PreFreelib 
        cuda_utilities_lib 
        CUDA::cusparse
        CUDA::cudart
    )
else()
    message(FATAL_ERROR "No test files found!")
endif()

# Add statistic executable
file(GLOB STATISTIC_FILES "${CUDA_TEST_PATH}/statistic.cu")
if(STATISTIC_FILES)
    add_executable(statistic ${STATISTIC_FILES})
    set_property(TARGET statistic PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(statistic PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )

    target_link_libraries(statistic PRIVATE 
        PreFreelib 
        cuda_utilities_lib 
        CUDA::cusparse
        CUDA::cudart
    )
    
    message(STATUS "Added statistic executable")
else()
    message(WARNING "No statistic files found!")
endif()