set(DIV_REGION_EXE_NAME divRegion)

# Full paths.
set(KERNEL_DIR \"${DIV_REGION_PATH}/${KERNEL_DIR}/\")
set(INCLUDE_TARGET_DIR "${CMAKE_BINARY_DIR}/${TESTS_DIR}/${DIV_REGION}/${INCLUDE_DIR}/")

# Src files.
file(GLOB DIV_REGION_FILE_LIST "${DIV_REGION_PATH}/${SRC_DIR}/main.cpp")

add_executable(${DIV_REGION_EXE_NAME} ${DIV_REGION_FILE_LIST})

# Include directories.
configure_file(${DIV_REGION_PATH}/${INCLUDE_DIR}/SystemConfiguration.h.cmake 
               ${INCLUDE_TARGET_DIR}/SystemConfiguration.h)
include_directories("${DIV_REGION_PATH}/${INCLUDE_DIR}" "${INCLUDE_TARGET_DIR}")

target_link_libraries(${DIV_REGION_EXE_NAME} ${OPENCL_LIBRARY_PATH} ${OPENCL_WRAPPER_LIB} ${BENCH_SUPPORT_LIB})
