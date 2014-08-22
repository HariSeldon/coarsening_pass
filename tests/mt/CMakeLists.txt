set(MT_EXE_NAME mt)

# Full paths.
set(KERNEL_DIR \"${MT_PATH}/${KERNEL_DIR}/\")
set(INCLUDE_TARGET_DIR "${CMAKE_BINARY_DIR}/${TESTS_DIR}/${MT}/${INCLUDE_DIR}/")

# Src files.
file(GLOB MT_FILE_LIST "${MT_PATH}/${SRC_DIR}/main.cpp")

add_executable(${MT_EXE_NAME} ${MT_FILE_LIST})

# Include directories.
configure_file(${MT_PATH}/${INCLUDE_DIR}/SystemConfiguration.h.cmake 
               ${INCLUDE_TARGET_DIR}/SystemConfiguration.h)
include_directories("${MT_PATH}/${INCLUDE_DIR}" "${INCLUDE_TARGET_DIR}")

target_link_libraries(${MT_EXE_NAME} ${OPENCL_LIBRARY_PATH} ${OPENCL_WRAPPER_LIB} ${BENCH_SUPPORT_LIB})
