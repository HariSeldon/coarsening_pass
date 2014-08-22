/**
 * atax.c: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <iostream>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "bench_support.h"
#include "MathUtils.h"
#include "SystemConfig.h"
#include "PolybenchUtils.h"

#define MAX_SOURCE_SIZE (0x100000)

/* Problem size. */
#define NX_DEFAULT 4096 
#define NY_DEFAULT 64

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 256
#define DIM_LOCAL_WORK_GROUP_Y 1

#ifndef M_PI
#define M_PI 3.14159
#endif

#if defined(cl_khr_fp64) // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64) // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

char str_temp[1024];

size_t NX = NX_DEFAULT;

cl_platform_id platform_id;
cl_device_id device_id;
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;
cl_context clGPUContext;
cl_kernel clKernel1;
cl_kernel clKernel2;
cl_command_queue clCommandQue;
cl_program clProgram;
cl_mem a_mem_obj;
cl_mem x_mem_obj;
cl_mem y_mem_obj;
cl_mem tmp_mem_obj;
FILE *fp;
char *source_str;
size_t source_size;

void compareResults(DATA_TYPE *z, DATA_TYPE *z_outputFromGpu) {
  int i, fail;
  fail = 0;

  for (i = 0; i < NY_DEFAULT; i++) {
    if (percentDiff(z[i], z_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
      fail++;
    }
  }

  assert(fail == 0 && "CPU - GPU Computation does not match!");
}

void read_cl_file() {
  // Load the kernel source code into the array source_str
  fp = fopen(KERNEL_PATH "/atax.cl", "r");
  if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
  }
  source_str = (char *)malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);
}

void init_array(DATA_TYPE *x, DATA_TYPE *A) {
  int i, j;

  for (int column = 0; column < NY_DEFAULT; ++column) {
    x[column] = random<DATA_TYPE>();

    for (int row = 0; row < NX; ++row) {
      A[row * NY_DEFAULT + column] = random<DATA_TYPE>();
    }

  }
}

void cl_mem_init(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *tmp) {
  a_mem_obj =
      clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE,
                     sizeof(DATA_TYPE) * NX * NY_DEFAULT, NULL, &errcode);
  x_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE,
                             sizeof(DATA_TYPE) * NY_DEFAULT, NULL, &errcode);
  y_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE,
                             sizeof(DATA_TYPE) * NY_DEFAULT, NULL, &errcode);
  tmp_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE,
                               sizeof(DATA_TYPE) * NX, NULL, &errcode);

  if (errcode != CL_SUCCESS)
    printf("Error in creating buffers\n");

  errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0,
                                 sizeof(DATA_TYPE) * NX * NY_DEFAULT, A, 0,
                                 NULL, NULL);
  errcode =
      clEnqueueWriteBuffer(clCommandQue, x_mem_obj, CL_TRUE, 0,
                           sizeof(DATA_TYPE) * NY_DEFAULT, x, 0, NULL, NULL);
  errcode =
      clEnqueueWriteBuffer(clCommandQue, y_mem_obj, CL_TRUE, 0,
                           sizeof(DATA_TYPE) * NY_DEFAULT, y, 0, NULL, NULL);
  errcode = clEnqueueWriteBuffer(clCommandQue, tmp_mem_obj, CL_TRUE, 0,
                                 sizeof(DATA_TYPE) * NX, tmp, 0, NULL, NULL);
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

  // Create the 1st OpenCL kernel
  clKernel1 = clCreateKernel(clProgram, "atax_kernel1", &errcode);
  if (errcode != CL_SUCCESS)
    printf("Error in creating kernel\n");

  // Create the 2nd OpenCL kernel
  clKernel2 = clCreateKernel(clProgram, "atax_kernel2", &errcode);
  if (errcode != CL_SUCCESS)
    printf("Error in creating kernel\n");
  clFinish(clCommandQue);
}

void cl_launch_kernel() {
  double t_start, t_end;

  int nx = NX;
  int ny = NY_DEFAULT;

  size_t oldLocalWorkSize[1], globalWorkSize[1];
  oldLocalWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
  globalWorkSize[0] = NX;

  ///////////////////////////////////////////////
  size_t localWorkSize[1];
  getNewSizes(NULL, oldLocalWorkSize, NULL, localWorkSize, "atax_kernel1", 1);
  ///////////////////////////////////////////////

  // Set the arguments of the kernel
  errcode = clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&a_mem_obj);
  errcode |= clSetKernelArg(clKernel1, 1, sizeof(cl_mem), (void *)&x_mem_obj);
  errcode |= clSetKernelArg(clKernel1, 2, sizeof(cl_mem), (void *)&tmp_mem_obj);
  errcode |= clSetKernelArg(clKernel1, 3, sizeof(int), (void *)&nx);
  errcode |= clSetKernelArg(clKernel1, 4, sizeof(int), (void *)&ny);
  if (errcode != CL_SUCCESS)
    printf("Error in seting arguments\n");

  // Execute the OpenCL kernel
  errcode =
      clEnqueueNDRangeKernel(clCommandQue, clKernel1, 1, NULL, globalWorkSize,
                             localWorkSize, 0, NULL, NULL);
  if (errcode != CL_SUCCESS)
    printf("Error in launching kernel\n");
//  clEnqueueBarrier(clCommandQue);
//
//  globalWorkSize[0] =
//      (size_t)ceil(((float)NY_DEFAULT) / ((float)DIM_LOCAL_WORK_GROUP_X)) *
//      DIM_LOCAL_WORK_GROUP_X;
//  globalWorkSize[1] = 1;

//  // Set the arguments of the kernel
//  errcode = clSetKernelArg(clKernel2, 0, sizeof(cl_mem), (void *)&a_mem_obj);
//  errcode |= clSetKernelArg(clKernel2, 1, sizeof(cl_mem), (void *)&y_mem_obj);
//  errcode |= clSetKernelArg(clKernel2, 2, sizeof(cl_mem), (void *)&tmp_mem_obj);
//  errcode |= clSetKernelArg(clKernel2, 3, sizeof(int), (void *)&nx);
//  errcode |= clSetKernelArg(clKernel2, 4, sizeof(int), (void *)&ny);
//  if (errcode != CL_SUCCESS)
//    printf("Error in seting arguments\n");
//  errcode =
//      clEnqueueNDRangeKernel(clCommandQue, clKernel2, 1, NULL, globalWorkSize,
//                             localWorkSize, 0, NULL, NULL);
//  if (errcode != CL_SUCCESS)
//    printf("Error in launching kernel\n");
  clFinish(clCommandQue);
}

void cl_clean_up() {
  // Clean up
  errcode = clFlush(clCommandQue);
  errcode = clFinish(clCommandQue);
  errcode = clReleaseKernel(clKernel1);
  errcode = clReleaseKernel(clKernel2);
  errcode = clReleaseProgram(clProgram);
  errcode = clReleaseMemObject(a_mem_obj);
  errcode = clReleaseMemObject(x_mem_obj);
  errcode = clReleaseMemObject(y_mem_obj);
  errcode = clReleaseMemObject(tmp_mem_obj);
  errcode = clReleaseCommandQueue(clCommandQue);
  errcode = clReleaseContext(clGPUContext);
  if (errcode != CL_SUCCESS)
    printf("Error in cleanup\n");
}

void atax_cpu(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *tmp,
              DATA_TYPE *result) {
  int i, j;

  char *reps = getEnvString("OCL_REPETITIONS");
  int intReps = 1;
  if (reps != NULL) {
    intReps = atoi(reps);
  }

  for (int row = 0; row < 32; row++) {
      tmp[row] = 0;
    for (int rep = 0; rep < intReps; ++rep) {
      for (int column = 0; column < NY_DEFAULT; column++) {
        tmp[row] += A[row * NY_DEFAULT + column] * x[column];
      }
    }

    assert(fabs(tmp[row] - result[row]) < 1 && "Error!");
  }

  std::cout << "Ok!\n";
}

int main(void) {
  double t_start, t_end;

  DATA_TYPE *A;
  DATA_TYPE *x;
  DATA_TYPE *y;
  DATA_TYPE *y_outputFromGpu;
  DATA_TYPE *tmp;

  /////////////////////////
  size_t oldSizes[1] = { NX };
  size_t newSizes[1];
  getNewSizes(oldSizes, NULL, newSizes, NULL, "atax_kernel1", 1);
  NX = newSizes[0];
  /////////////////////////

  A = (DATA_TYPE *)malloc(NX * NY_DEFAULT * sizeof(DATA_TYPE));
  x = (DATA_TYPE *)malloc(NY_DEFAULT * sizeof(DATA_TYPE));
  tmp = (DATA_TYPE *)malloc(NX * sizeof(DATA_TYPE));

  init_array(x, A);
  read_cl_file();
  cl_initialization(device_id, clGPUContext, clCommandQue);
  cl_mem_init(A, x, y, tmp);
  cl_load_prog();

  cl_launch_kernel();

  errcode = clEnqueueReadBuffer(clCommandQue, tmp_mem_obj, CL_TRUE, 0,
                                NX_DEFAULT * sizeof(DATA_TYPE), tmp,
                                0, NULL, NULL);
  if (errcode != CL_SUCCESS)
    printf("Error in reading GPU mem\n");

  atax_cpu(A, x, y, tmp, tmp);
  cl_clean_up();

  free(A);
  free(x);
  free(tmp);

  return 0;
}
