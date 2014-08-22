set(SRC_DIR "src")
set(KERNEL_DIR "kernels")
set(INCLUDE_DIR "include")
set(OPENCL_WRAPPER_HEADER "${PROJECT_SOURCE_DIR}/${OPENCL_TOOLS_DIR}/${OPENCL_WRAPPER_DIR}/include")
set(OPENCL_BENCH_SUPPORT "${PROJECT_SOURCE_DIR}/${OPENCL_TOOLS_DIR}/${BENCH_SUPPORT_DIR}/include")

include_directories("${OPENCL_INCLUDE_PATH}" "${OPENCL_BENCH_SUPPORT}" "${OPENCL_WRAPPER_HEADER}")

# List test directories.
set(MM "mm")
set(MT "mt")
set(MV "mv")
set(MEMSET "memset")
set(MEMCPY "memcpy")
set(POLYBENCH "polybench")
set(DIV_REGION "divRegion")

set(MM_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${MM})
set(MT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${MT})
set(MV_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${MV})
set(MEMCPY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${MEMCPY})
set(MEMSET_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${MEMSET})
set(DIV_REGION_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${DIV_REGION})
set(POLYBENCH_TEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${POLYBENCH})

################################################################################
# List of testing directories.

add_subdirectory(${MM_PATH})
add_subdirectory(${MT_PATH})
add_subdirectory(${MV_PATH})
add_subdirectory(${MEMCPY_PATH})
add_subdirectory(${MEMSET_PATH})
add_subdirectory(${DIV_REGION})
add_subdirectory(${POLYBENCH_TEST_PATH})
