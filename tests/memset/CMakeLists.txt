set(MEMSET_EXE_NAME memset)

# Full paths.
set(KERNEL_DIR \"${MEMSET_PATH}/${KERNEL_DIR}/\")
set(INCLUDE_TARGET_DIR "${CMAKE_BINARY_DIR}/${TESTS_DIR}/${MEMSET}/${INCLUDE_DIR}/")

# Src files.
file(GLOB MEMSET_FILE_LIST "${MEMSET_PATH}/${SRC_DIR}/main.cpp")

add_executable(${MEMSET_EXE_NAME} ${MEMSET_FILE_LIST})

# Include directories.
configure_file(${MEMSET_PATH}/${INCLUDE_DIR}/SystemConfiguration.h.cmake 
               ${INCLUDE_TARGET_DIR}/SystemConfiguration.h)
include_directories("${MEMSET_PATH}/${INCLUDE_DIR}" "${INCLUDE_TARGET_DIR}")

target_link_libraries(${MEMSET_EXE_NAME} ${OPENCL_LIBRARY_PATH} ${OPENCL_WRAPPER_LIB} ${BENCH_SUPPORT_LIB})
