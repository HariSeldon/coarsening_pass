set(AXTOR_LIB "axtorwrapper")
set(OCL_LIB "oclwrapper")

set(INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/include/")

set(AXTOR_FILE_LIST "${CMAKE_CURRENT_SOURCE_DIR}/src/Utils.cpp"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/AxtorWrapper.cpp")

set(OCL_FILE_LIST "${CMAKE_CURRENT_SOURCE_DIR}/src/OCLWrapper.cpp"
                  "${CMAKE_CURRENT_SOURCE_DIR}/src/Utils.cpp")

include_directories(${INCLUDE_PATH} ${OPENCL_INCLUDE_PATH}) 

add_library(${AXTOR_LIB} SHARED ${AXTOR_FILE_LIST})
add_library(${OCL_LIB} SHARED ${OCL_FILE_LIST})
