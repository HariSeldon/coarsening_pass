set(MM_EXE_NAME mm)

# Full paths.
set(KERNEL_DIR \"${MM_PATH}/${KERNEL_DIR}/\")
set(INCLUDE_TARGET_DIR "${CMAKE_BINARY_DIR}/${TESTS_DIR}/${MM}/${INCLUDE_DIR}/")

# Src files.
file(GLOB MM_FILE_LIST "${MM_PATH}/${SRC_DIR}/main.cpp")

add_executable(${MM_EXE_NAME} ${MM_FILE_LIST})

# Include directories.
configure_file(${MM_PATH}/${INCLUDE_DIR}/SystemConfiguration.h.cmake 
               ${INCLUDE_TARGET_DIR}/SystemConfiguration.h)
include_directories("${MM_PATH}/${INCLUDE_DIR}" "${INCLUDE_TARGET_DIR}")

target_link_libraries(${MM_EXE_NAME} ${OPENCL_LIBRARY_PATH} ${OPENCL_WRAPPER_LIB} ${BENCH_SUPPORT_LIB})
