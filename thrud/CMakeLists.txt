set(THRUD "Thrud")
set(LIB_DIR "lib")
set(INCLUDE_DIR "include")

# Find llvm.
set(LLVM_DIR "~/build/llvm/share/llvm/cmake/")
find_package(LLVM REQUIRED)

# Include dirs.
include_directories(${INCLUDE_DIR})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})

# Link dirs.
link_directories(${LLVM_LIBRARY_DIRS})

# Definitions.
add_definitions("-D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS")

# Subdir.
add_subdirectory(${LIB_DIR})
