file(GLOB SRC_FILE_LIST "*.cpp")
add_library(${THRUD} MODULE ${SRC_FILE_LIST})
set_target_properties(${THRUD} PROPERTIES COMPILE_FLAGS "-fno-rtti -fPIC")

install_targets("/${LIB_DIR}/" ${THRUD})
