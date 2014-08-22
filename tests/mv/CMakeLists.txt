set(MV_EXE_NAME mv)

# Full paths.
set(KERNEL_DIR \"${MV_PATH}/${KERNEL_DIR}/\")
set(INCLUDE_TARGET_DIR "${CMAKE_BINARY_DIR}/${TESTS_DIR}/${MV}/${INCLUDE_DIR}/")

# Src files.
file(GLOB MV_FILE_LIST "${MV_PATH}/${SRC_DIR}/main.cpp")

add_executable(${MV_EXE_NAME} ${MV_FILE_LIST})

# Include directories.
configure_file(${MV_PATH}/${INCLUDE_DIR}/SystemConfiguration.h.cmake 
               ${INCLUDE_TARGET_DIR}/SystemConfiguration.h)
include_directories("${MV_PATH}/${INCLUDE_DIR}" "${INCLUDE_TARGET_DIR}")

target_link_libraries(${MV_EXE_NAME} ${OPENCL_LIBRARY_PATH} ${OPENCL_WRAPPER_LIB} ${BENCH_SUPPORT_LIB})
