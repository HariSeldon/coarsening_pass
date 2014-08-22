#! /bin/bash

CLANG=clang
OPT=opt
LLVM_DIS=llvm-dis

INPUT_FILE=$1
KERNEL_NAME=$2

OCLDEF=$HOME/src/coarsening_pass/thrud/include/opencl_spir.h

$CLANG -x cl \
       -O0 \
       -target spir \
       -include ${OCLDEF} \
       ${INPUT_FILE} \
       -S -emit-llvm -fno-builtin -o - |
$OPT -instnamer \
     -mem2reg \
     -o - |
$LLVM_DIS -o -  
