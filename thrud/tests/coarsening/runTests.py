#! /usr/bin/python

import itertools;
import os;
import subprocess;
import time;

tests = {
"memset.cl" : ["memset1", "memset2"], 
"mm.cl" : ["mm"], 
"mt.cl" : ["mt"],
"2DConvolution.cl" : ["Convolution2D_kernel"],
"2mm.cl" : ["mm2_kernel1"],
"3DConvolution.cl" : ["Convolution3D_kernel"],
"3mm.cl" : ["mm3_kernel1"],
"atax.cl" : ["atax_kernel1", "atax_kernel2"],
"bicg.cl" : ["bicgKernel1"],
"correlation.cl" : ["mean_kernel", "std_kernel", "reduce_kernel", "corr_kernel"],
"covariance.cl" : ["mean_kernel", "reduce_kernel", "covar_kernel"],
"fdtd2d.cl" : ["fdtd_kernel1", "fdtd_kernel2", "fdtd_kernel3"],
"gemm.cl" : ["gemm"],
"gesummv.cl" : ["gesummv_kernel"],
"gramschmidt.cl" : ["gramschmidt_kernel1", "gramschmidt_kernel2", "gramschmidt_kernel3"],
"mm2metersKernel.cl" : ["mm2metersKernel"],
"mvt.cl" : ["mvt_kernel1"],
"syr2k.cl" : ["syr2k_kernel"],
"syrk.cl" : ["syrk_kernel"],
"spmv.cl" : ["spmv_jds_naive"],
"stencil.cl" : ["naive_kernel"]
}; 

HOME = os.environ["HOME"]; 
CLANG = "clang";
OPT = "opt";
LIB_THRUD = os.path.join(HOME, "root", "lib", "libThrud.so");
OCLDEF = os.path.join(HOME, "src", "coarsening_pass", "thrud", "include", "opencl_spir.h");
OPTIMIZATION = "-O0";

#-------------------------------------------------------------------------------
def printRed(message):
  print "\033[1;31m%s\033[1;m" % message

#-------------------------------------------------------------------------------
def printGreen(message):
  print "\033[1;32m%s\033[1;m" % message;

#-------------------------------------------------------------------------------
def runCommand(arguments, toStdin = None):
  commandOutput = None;
  runProcess = subprocess.Popen(arguments, stdin = subprocess.PIPE,
                                           stdout = subprocess.PIPE,
                                           stderr = subprocess.PIPE);
  commandOutput = runProcess.communicate(toStdin);
  runReturnCode = runProcess.poll();
  return (runReturnCode, commandOutput[0], commandOutput[1]);

#-------------------------------------------------------------------------------
def runTest(fileName, kernelName, cd, cf, st):
  fileName = os.path.join("kernels", fileName);
  clangCommand = [CLANG, "-x", "cl", "-target", "spir", "-include", OCLDEF, 
                  OPTIMIZATION, fileName, "-S", "-emit-llvm", "-fno-builtin", 
                  "-o", "-"];
  optCommand = [OPT, "-mem2reg", "-instnamer", "-load", LIB_THRUD, 
                "-structurizecfg", "-be", "-tc", 
                "-coarsening-factor", cf, 
                "-coarsening-direction", cd, 
                "-coarsening-stride", st,
                "-div-region-mgt", "classic",
                "-kernel-name", kernelName,
                "-o", "/dev/null"];
  clangResult = runCommand(clangCommand);
  if(clangResult[0] == 0):
    llvmModule = clangResult[1];
    optResult = runCommand(optCommand, llvmModule);
    if(optResult[0] == 0):
      printGreen(" ".join([fileName, kernelName, cd, cf, st, "Ok"])); 
    else:
      print(optResult[2]);
      printRed(" ".join([fileName, kernelName, cd, cf, st, "opt Failed!"])); 
  else:
    print(clangResult[2]);
    printRed(" ".join([fileName, kernelName, cd, cf, st, "clang Failed!"])); 

def main():
  directions = ["0", "1"];
  factors = ["1", "2", "4", "8", "16", "32"];
  strides = ["1", "2", "4", "8", "16", "32"]; 
  factors = ["2"];
  strides = ["2"]; 

  configs = itertools.product(directions, factors, strides);
  configs = [x for x in configs];

  for test in tests:
    kernels = tests[test];
    for kernel in kernels:
      for config in configs:
        runTest(test, kernel, *config);      

main();
