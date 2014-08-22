set(INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/include/")

file(GLOB SRC_FILE_LIST "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# Include directories.
include_directories(${INCLUDE_PATH})

add_library(${BENCH_SUPPORT_LIB} SHARED ${SRC_FILE_LIST})
