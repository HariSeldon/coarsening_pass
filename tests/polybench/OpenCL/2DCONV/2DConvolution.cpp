/**
 * 2DConvolution.c: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "bench_support.h"
#include "MathUtils.h"
#include "SystemConfig.h"
#include "PolybenchUtils.h"

//define the error threshold for the results "not matching"

#define MAX_SOURCE_SIZE (0x100000)

/* Problem size */
#define NI_DEFAULT 4096
#define NJ_DEFAULT 4096

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 32
#define DIM_LOCAL_WORK_GROUP_Y 8

#if defined(cl_khr_fp64) // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64) // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

cl_platform_id platform_id;
cl_device_id device_id;
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;
cl_context clGPUContext;
cl_kernel clKernel;
cl_command_queue clCommandQue;
cl_program clProgram;
cl_mem a_mem_obj;
cl_mem b_mem_obj;
cl_mem c_mem_obj;
FILE *fp;
char *source_str;
size_t source_size;
unsigned int NI = NI_DEFAULT;
unsigned int NJ = NJ_DEFAULT;

void read_cl_file() {
  // Load the kernel source code into the array source_str
  fp = fopen(KERNEL_PATH "/2DConvolution.cl", "r");
  if (!fp) {
    fprintf(stdout, "Failed to load kernel.\n");
    exit(1);
  }
  source_str = (char *)malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);
}

void init(DATA_TYPE *A) {
  int i, j;

  for (i = 0; i < NI; ++i) {
    for (j = 0; j < NJ; ++j) {
      A[i * NJ + j] = random<float>();
    }
  }
}

void cl_mem_init(DATA_TYPE *A) {
  a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_ONLY,
                             sizeof(DATA_TYPE) * NI * NJ, NULL, &errcode);
  b_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE,
                             sizeof(DATA_TYPE) * NI * NJ, NULL, &errcode);

  if (errcode != CL_SUCCESS)
    printf("Error in creating buffers\n");

  errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0,
                                 sizeof(DATA_TYPE) * NI * NJ, A, 0, NULL, NULL);
  if (errcode != CL_SUCCESS)
    printf("Error in writing buffers\n");
}

void cl_load_prog() {
  // Create a program from the kernel source
  clProgram =
      clCreateProgramWithSource(clGPUContext, 1, (const char **)&source_str,
                                (const size_t *)&source_size, &errcode);

  if (errcode != CL_SUCCESS)
    printf("Error in creating program\n");

  // Build the program
  errcode = clBuildProgram(clProgram, 1, &device_id, NULL, NULL, NULL);
  if (errcode != CL_SUCCESS)
    printf("Error in building program\n");

  // Create the OpenCL kernel
  clKernel = clCreateKernel(clProgram, "Convolution2D_kernel", &errcode);
  if (errcode != CL_SUCCESS)
    printf("Error in creating kernel\n");
  clFinish(clCommandQue);
}

void cl_launch_kernel() {
  int ni = NI;
  int nj = NJ;

  size_t oldLocalWorkSize[2], globalWorkSize[2];
  oldLocalWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
  oldLocalWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
  globalWorkSize[0] = NI;
  globalWorkSize[1] = NJ;

  ///////////////////////////////////////////////
  size_t localWorkSize[2];
  getNewSizes(NULL, oldLocalWorkSize, NULL, localWorkSize,
              "Convolution2D_kernel", 2);
  ///////////////////////////////////////////////

  // Set the arguments of the kernel
  errcode = clSetKernelArg(clKernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
  errcode |= clSetKernelArg(clKernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
  errcode = clSetKernelArg(clKernel, 2, sizeof(int), &ni);
  errcode |= clSetKernelArg(clKernel, 3, sizeof(int), &nj);

  if (errcode != CL_SUCCESS)
    printf("Error in seting arguments\n");
  // Execute the OpenCL kernel
  errcode =
      clEnqueueNDRangeKernel(clCommandQue, clKernel, 2, NULL, globalWorkSize,
                             localWorkSize, 0, NULL, NULL);

  if (errcode != CL_SUCCESS)
    printf("Error in launching kernel\n");
  clFinish(clCommandQue);
}

void cl_clean_up() {
  // Clean up
  errcode = clFlush(clCommandQue);
  errcode = clFinish(clCommandQue);
  errcode = clReleaseKernel(clKernel);
  errcode = clReleaseProgram(clProgram);
  errcode = clReleaseMemObject(a_mem_obj);
  errcode = clReleaseMemObject(b_mem_obj);
  errcode = clReleaseCommandQueue(clCommandQue);
  errcode = clReleaseContext(clGPUContext);
  if (errcode != CL_SUCCESS)
    printf("Error in cleanup\n");
}

void conv2D(DATA_TYPE *A, DATA_TYPE *B_outputFromGpu) {
  int i, j;
  DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

  c11 = +0.2;
  c21 = +0.5;
  c31 = -0.8;
  c12 = -0.3;
  c22 = +0.6;
  c32 = -0.9;
  c13 = +0.4;
  c23 = +0.7;
  c33 = +0.10;

  for (i = 1; i < NI - 1; ++i) {
    for (j = 1; j < NJ - 1; ++j) {
      DATA_TYPE tmp =
          c11 * A[(i - 1) * NJ + (j - 1)] + c12 * A[(i + 0) * NJ + (j - 1)] +
          c13 * A[(i + 1) * NJ + (j - 1)] + c21 * A[(i - 1) * NJ + (j + 0)] +
          c22 * A[(i + 0) * NJ + (j + 0)] + c23 * A[(i + 1) * NJ + (j + 0)] +
          c31 * A[(i - 1) * NJ + (j + 1)] + c32 * A[(i + 0) * NJ + (j + 1)] +
          c33 * A[(i + 1) * NJ + (j + 1)];
      
      assert(fabs(tmp - B_outputFromGpu[i * NJ + j]) < 0.01f && "Error in the computation");
    }
  }
  std::cout << "Ok!\n";
}

int main(int argc, char *argv[]) {

  DATA_TYPE *A;
  DATA_TYPE *B;
  DATA_TYPE *B_outputFromGpu;

  /////////////////////////
  size_t oldSizes[2] = { NI, NJ };
  size_t newSizes[2];
  getNewSizes(oldSizes, NULL, newSizes, NULL, "Convolution2D_kernel", 2);
  NI = newSizes[0];
  NJ = newSizes[1];
  /////////////////////////

  A = (DATA_TYPE *)malloc(NI * NJ * sizeof(DATA_TYPE));
  B = (DATA_TYPE *)malloc(NI * NJ * sizeof(DATA_TYPE));
  B_outputFromGpu = (DATA_TYPE *)malloc(NI * NJ * sizeof(DATA_TYPE));

  init(A);

  read_cl_file();
  cl_initialization(device_id, clGPUContext, clCommandQue);
  cl_mem_init(A);
  cl_load_prog();

  cl_launch_kernel();

  errcode = clEnqueueReadBuffer(clCommandQue, b_mem_obj, CL_TRUE, 0,
                                NI * NJ * sizeof(DATA_TYPE), B_outputFromGpu, 0,
                                NULL, NULL);
  if (errcode != CL_SUCCESS)
    printf("Error in reading GPU mem\n");

  conv2D(A, B_outputFromGpu);

  free(A);
  free(B);
  free(B_outputFromGpu);

  cl_clean_up();
  return 0;
}
