cmake_minimum_required(VERSION 2.8.8)

project("coarsening_pass")

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
add_definitions("-std=c++11")

# Directories.
set(THRUD_DIR "thrud")
set(TESTS_DIR "tests")
set(OPENCL_TOOLS_DIR "opencl_tools")

# OpenCL tools.
set(BENCH_SUPPORT_LIB "bench_support")
set(BENCH_SUPPORT_DIR "bench_support")

set(FUNCTION_OVERLOAD_DIR "function_overload")

set(OPENCL_WRAPPER_DIR "opencl_wrapper")
set(OPENCL_WRAPPER_LIB "OpenCLWrapper")

# LLVM.
# Set this to your LLVM cmake directory.
set(LLVM_DIR "~/build/llvm/share/llvm/cmake/")

# OpenCL.
# Set this to the directory containing cl/cl.h.
set(OPENCL_INCLUDE_PATH "/opt/AMDAPP/include")
find_library(OPENCL_LIBRARY_PATH OpenCL
             "/opt/AMDAPP/lib/x86_64/"
             DOC "The OpenCL Library")
if(EXISTS ${OPENCL_LIBRARY_PATH})
  message(STATUS "Looking for OPENCL: found")
else(EXISTS ${OPENCL_LIBRARY_PATH})
  message(FATAL_ERROR "Looking for OPENCL: not found")
endif(EXISTS ${OPENCL_LIBRARY_PATH})

add_subdirectory(${THRUD_DIR})
add_subdirectory(${OPENCL_TOOLS_DIR})
add_subdirectory(${TESTS_DIR})
