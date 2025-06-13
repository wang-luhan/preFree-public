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

option(ENABLE_CUDA_MEMCHECK "Enable cuda-memcheck support" OFF)
option(USE_FP64 "Enable fp64 support" ON)

if(ENABLE_CUDA_MEMCHECK)
    message(STATUS "cuda-memcheck is enabled")
    set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -O0")
else()
    message(STATUS "cuda-memcheck is disabled")
endif()

if(USE_FP64)
    message(STATUS "FP64 is enabled")
    add_definitions(-Dfp64)
else()
    message(STATUS "FP64 is disabled")
endif()

set(the_description "cuda")  
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

# Find cuSPARSE library with multiple search strategies
find_library(CUSPARSE_LIBRARY 
    NAMES cusparse libcusparse.so libcusparse
    PATHS 
        "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib64"
        "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib"
        "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../targets/x86_64-linux/lib"
        "/usr/local/cuda/lib64"
        "/usr/local/cuda/lib"
        "/usr/local/cuda/targets/x86_64-linux/lib"
        "$ENV{CUDA_PATH}/lib64"
        "$ENV{CUDA_PATH}/lib"
        "$ENV{CUDA_HOME}/lib64"
        "$ENV{CUDA_HOME}/lib"
)


if(NOT CUSPARSE_LIBRARY)
    find_library(CUSPARSE_LIBRARY 
        NAMES cusparse libcusparse.so libcusparse
        PATHS /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib
    )
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)

if(CUSPARSE_LIBRARY)
    message(STATUS "cuSPARSE library found: ${CUSPARSE_LIBRARY}")
else()
    message(STATUS "Searching for cuSPARSE in system paths...")
    execute_process(
        COMMAND find /usr -name "*cusparse*" 2>/dev/null
        OUTPUT_VARIABLE CUSPARSE_SEARCH_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(CUSPARSE_SEARCH_RESULT)
        message(STATUS "Found cuSPARSE files: ${CUSPARSE_SEARCH_RESULT}")
    endif()
    message(FATAL_ERROR "cuSPARSE library not found! Please install CUDA toolkit with cuSPARSE.")
endif()

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
target_link_libraries(cuda_utilities_lib ${CUSPARSE_LIBRARY})

file(GLOB CUDA_TEST_FILES "${CUDA_TEST_PATH}/test_cdspmv.cpp")
message(STATUS "CUDA_TEST_FILES: ${CUDA_TEST_FILES}")

if(CUDA_TEST_FILES)
    add_executable(cuda_perftest ${CUDA_TEST_FILES})
    target_link_libraries(cuda_perftest PreFreelib cuda_utilities_lib ${CUSPARSE_LIBRARY})
    target_include_directories(cuda_perftest PRIVATE ${PROJECT_SOURCE_DIR}/include)
    
    # Link CUDA runtime
    if(TARGET CUDA::cudart)
        target_link_libraries(cuda_perftest CUDA::cudart)
    endif()
else()
    message(FATAL_ERROR "No test files found!")
endif()