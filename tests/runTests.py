#! /usr/bin/python

import itertools;
import os;
import subprocess;
import time;

tests = {
"memset/memset" : ["memset1", "memset2"], 
"memcpy/memcpy" : ["rmrrmw", "cmrcmw"],
"mm/mm" : ["mm"], 
"mt/mt" : ["mt"],
"mv/mv" : ["MatVecMulUncoalesced0", "MatVecMulUncoalesced1", "MatVecMulCoalesced0"],
"divRegion/divRegion" : ["divRegion"], 
"polybench/OpenCL/2DCONV/2DCONV" : ["Convolution2D_kernel"],
"polybench/OpenCL/2MM/2MM" : ["mm2_kernel1"],
"polybench/OpenCL/3DCONV/3DCONV" : ["Convolution3D_kernel"],
"polybench/OpenCL/3MM/3MM" : ["mm3_kernel1"],
"polybench/OpenCL/ATAX/ATAX" : ["atax_kernel1", "atax_kernel2"],
"polybench/OpenCL/BICG/BICG" : ["bicgKernel1"],
"polybench/OpenCL/CORR/CORR" : ["mean_kernel", "std_kernel", "reduce_kernel"],
"polybench/OpenCL/COVAR/COVAR" : ["mean_kernel", "reduce_kernel", "covar_kernel"],
"polybench/OpenCL/FDTD-2D/FDTD-2D" : ["fdtd_kernel1", "fdtd_kernel2", "fdtd_kernel3"],
"polybench/OpenCL/GEMM/GEMM" : ["gemm"],
"polybench/OpenCL/GESUMMV/GESUMMV" : ["gesummv_kernel"],
"polybench/OpenCL/GRAMSCHM/GRAMSCHM" : ["gramschmidt_kernel1", "gramschmidt_kernel2", "gramschmidt_kernel3"],
"polybench/OpenCL/MVT/MVT" : ["mvt_kernel1"],
"polybench/OpenCL/SYR2K/SYR2K" : ["syr2k_kernel"],
"polybench/OpenCL/SYRK/SYRK" : ["syrk_kernel"]
}; 

HOME = os.environ["HOME"]; 
CLANG = "clang";
OPT = "opt";
# Replace with Thrud path.
LIB_THRUD = os.path.join(HOME, "root", "lib", "libThrud.so");
OCL_HEADER = os.path.join(HOME, "src", "coarsening_pass", "thrud", "include", "opencl_spir.h");
OPTIMIZATION = "-O0";
# Replace with libaxtorpath path.
LD_PRELOAD = os.path.join(HOME, "build", "coarsening_pass", "opencl_tools", "function_overload", "libaxtorwrapper.so");
TC_COMPILE_LINE = "-mem2reg -load " + LIB_THRUD + " -structurizecfg -instnamer -be -tc -coarsening-factor %s -coarsening-direction %s -coarsening-stride %s -div-region-mgt classic -kernel-name %s -simplifycfg";
PREFIX = os.path.join(HOME, "build", "coarsening_pass", "tests");

#-------------------------------------------------------------------------------
def printRed(message):
  print "\033[1;31m%s\033[1;m" % message

#-------------------------------------------------------------------------------
def printGreen(message):
  print "\033[1;32m%s\033[1;m" % message;

#-------------------------------------------------------------------------------
def runCommand(arguments):
  WAITING_TIME = 30;
  runProcess = subprocess.Popen(arguments, stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE);
  runPid = runProcess.pid;
  counter = 0;
  returnCode = None;

  # Manage the case in which the run hangs.
  while(counter < WAITING_TIME and returnCode == None):
    counter += 1;
    time.sleep(1);
    returnCode = runProcess.poll();

  if(returnCode == None):
    runProcess.kill();
    return (-1, "", "Time expired!");

  commandOutput = runProcess.communicate();

  print(commandOutput[0]);
  print(commandOutput[1]);

  return (returnCode, commandOutput[0], commandOutput[1]);

#-------------------------------------------------------------------------------
def runTest(command, kernelName, cd, cf, st):
  command = [os.path.join(PREFIX, command), kernelName];

  os.environ["OCL_HEADER"] = OCL_HEADER;
  os.environ["TC_KERNEL_NAME"] = kernelName;
  os.environ["LD_PRELOAD"] = LD_PRELOAD;
  os.environ["OCL_COMPILER_OPTIONS"] = TC_COMPILE_LINE % \
    (cf, cd, st, kernelName);
  
  print(command);
  result = runCommand(command);
  if(result[0] == 0):
    printGreen(" ".join([kernelName, cd, cf, st, "Ok!"]));
    return 0;
  else:
    print(result[2]);
    printRed(" ".join([kernelName, cd, cf, st, "Failure"]));
    return 1;

def main():
  directions = ["0", "1"];
  factors = ["2", "4"];
  strides = ["1", "2", "32"]; 

  configs = itertools.product(directions, factors, strides);
  configs = [x for x in configs];

  counter = 0;
  failures = 0;

  for test in tests:
    kernels = tests[test];
    for kernel in kernels:
      for config in configs:
        failure = runTest(test, kernel, *config);      
        failures += failure;
        counter += 1;

  print("#################################");
  print(str(failures) + " failures out of " + str(counter));

main();
