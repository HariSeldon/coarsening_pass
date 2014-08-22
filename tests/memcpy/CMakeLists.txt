set(MEMCPY_EXE_NAME memcpy)

# Full paths.
set(KERNEL_DIR \"${MEMCPY_PATH}/${KERNEL_DIR}/\")
set(INCLUDE_TARGET_DIR "${CMAKE_BINARY_DIR}/${TESTS_DIR}/${MEMCPY}/${INCLUDE_DIR}/")

# Src files.
file(GLOB MEMCPY_FILE_LIST "${MEMCPY_PATH}/${SRC_DIR}/main.cpp")

add_executable(${MEMCPY_EXE_NAME} ${MEMCPY_FILE_LIST})

# Include directories.
configure_file(${MEMCPY_PATH}/${INCLUDE_DIR}/SystemConfiguration.h.cmake 
               ${INCLUDE_TARGET_DIR}/SystemConfiguration.h)
include_directories("${MEMCPY_PATH}/${INCLUDE_DIR}" "${INCLUDE_TARGET_DIR}")

target_link_libraries(${MEMCPY_EXE_NAME} ${OPENCL_LIBRARY_PATH} ${OPENCL_WRAPPER_LIB} ${BENCH_SUPPORT_LIB})
