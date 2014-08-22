/**
 * gesummv.c: This file is part of the PolyBench/GPU 1.0 test suite.
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

//define the error threshold for the results "not matching"

#define MAX_SOURCE_SIZE (0x100000)

/* Problem size */
#define N_DEFAULT 4096

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 256

#if defined(cl_khr_fp64) // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64) // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

char str_temp[1024];

DATA_TYPE ALPHA = 1;
DATA_TYPE BETA = 1;

cl_platform_id platform_id;
cl_device_id device_id;
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;
cl_context clGPUContext;
cl_kernel clKernel1;
cl_kernel clKernel2;
cl_kernel clKernel3;
cl_command_queue clCommandQue;
cl_program clProgram;

cl_mem a_mem_obj;
cl_mem b_mem_obj;
cl_mem x_mem_obj;
cl_mem y_mem_obj;
cl_mem tmp_mem_obj;

FILE *fp;
char *source_str;
size_t source_size;

size_t N = N_DEFAULT;

void init(DATA_TYPE *A, DATA_TYPE *x) {
  int i, j;

  for (i = 0; i < N; i++) {
    x[i] = ((DATA_TYPE)i) / N;

    for (j = 0; j < N; j++) {
      A[i * N + j] = random<DATA_TYPE>();
    }
  }
}

void read_cl_file() {
  // Load the kernel source code into the array source_str
  fp = fopen(KERNEL_PATH "/gesummv.cl", "r");
  if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
  }
  source_str = (char *)malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);
}

void gesummv(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x, DATA_TYPE *y,
             DATA_TYPE *tmp, DATA_TYPE *result) {
  int i, j;

  char *reps = getEnvString("OCL_REPETITIONS");
  int intReps = 1;
  if (reps != NULL) {
    intReps = atoi(reps);
  }

  for (i = 0; i < N; i++) {
    tmp[i] = 0;
    y[i] = 0;
    for (int rep = 0; rep < intReps; ++rep) {
      for (j = 0; j < N; j++) {
        tmp[i] = A[i * N + j] * x[j] + tmp[i];
        y[i] = B[i * N + j] * x[j] + y[i];
      }

      y[i] = ALPHA * tmp[i] + BETA * y[i];
    }

    assert(fabs(y[i] - result[i]) / y[i] < 0.001 && "Error!");
  }

  std::cout << "Ok!\n";
}

void cl_mem_init(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x, DATA_TYPE *y,
                 DATA_TYPE *tmp) {
  a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE,
                             sizeof(DATA_TYPE) * N * N, NULL, &errcode);
  b_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE,
                             sizeof(DATA_TYPE) * N * N, NULL, &errcode);
  x_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE,
                             sizeof(DATA_TYPE) * N, NULL, &errcode);
  y_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE,
                             sizeof(DATA_TYPE) * N, NULL, &errcode);
  tmp_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE,
                               sizeof(DATA_TYPE) * N, NULL, &errcode);

  if (errcode != CL_SUCCESS)
    printf("Error in creating buffers\n");

  errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0,
                                 sizeof(DATA_TYPE) * N * N, A, 0, NULL, NULL);
  errcode = clEnqueueWriteBuffer(clCommandQue, b_mem_obj, CL_TRUE, 0,
                                 sizeof(DATA_TYPE) * N * N, B, 0, NULL, NULL);
  errcode = clEnqueueWriteBuffer(clCommandQue, x_mem_obj, CL_TRUE, 0,
                                 sizeof(DATA_TYPE) * N, x, 0, NULL, NULL);
  errcode = clEnqueueWriteBuffer(clCommandQue, y_mem_obj, CL_TRUE, 0,
                                 sizeof(DATA_TYPE) * N, y, 0, NULL, NULL);
  errcode = clEnqueueWriteBuffer(clCommandQue, tmp_mem_obj, CL_TRUE, 0,
                                 sizeof(DATA_TYPE) * N, tmp, 0, NULL, NULL);
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
  clKernel1 = clCreateKernel(clProgram, "gesummv_kernel", &errcode);
  if (errcode != CL_SUCCESS)
    printf("Error in creating kernel1\n");
  clFinish(clCommandQue);
}

void cl_launch_kernel() {
  int n = N;

  size_t oldLocalWorkSize[1], globalWorkSize[1];
  oldLocalWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
  globalWorkSize[0] = N;

  ///////////////////////////////////////////////
  size_t localWorkSize[1];
  getNewSizes(NULL, oldLocalWorkSize, NULL, localWorkSize,
              "gesummv_kernel", 1);
  ///////////////////////////////////////////////

  // Set the arguments of the kernel
  errcode = clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&a_mem_obj);
  errcode |= clSetKernelArg(clKernel1, 1, sizeof(cl_mem), (void *)&b_mem_obj);
  errcode |= clSetKernelArg(clKernel1, 2, sizeof(cl_mem), (void *)&x_mem_obj);
  errcode |= clSetKernelArg(clKernel1, 3, sizeof(cl_mem), (void *)&y_mem_obj);
  errcode |= clSetKernelArg(clKernel1, 4, sizeof(cl_mem), (void *)&tmp_mem_obj);
  errcode |= clSetKernelArg(clKernel1, 5, sizeof(DATA_TYPE), (void *)&ALPHA);
  errcode |= clSetKernelArg(clKernel1, 6, sizeof(DATA_TYPE), (void *)&BETA);
  errcode |= clSetKernelArg(clKernel1, 7, sizeof(int), (void *)&n);

  if (errcode != CL_SUCCESS)
    printf("Error in seting arguments1\n");

  // Execute the OpenCL kernel
  errcode =
      clEnqueueNDRangeKernel(clCommandQue, clKernel1, 1, NULL, globalWorkSize,
                             localWorkSize, 0, NULL, NULL);
  if (errcode != CL_SUCCESS)
    printf("Error in launching kernel1\n");
  clFinish(clCommandQue);
}

void cl_clean_up() {
  // Clean up
  errcode = clFlush(clCommandQue);
  errcode = clFinish(clCommandQue);
  errcode = clReleaseKernel(clKernel1);
  errcode = clReleaseProgram(clProgram);
  errcode = clReleaseMemObject(a_mem_obj);
  errcode = clReleaseMemObject(b_mem_obj);
  errcode = clReleaseMemObject(x_mem_obj);
  errcode = clReleaseCommandQueue(clCommandQue);
  errcode = clReleaseContext(clGPUContext);
  if (errcode != CL_SUCCESS)
    printf("Error in cleanup\n");
}

int main(void) {
  DATA_TYPE *A;
  DATA_TYPE *B;
  DATA_TYPE *x;
  DATA_TYPE *y;
  DATA_TYPE *y_outputFromGpu;
  DATA_TYPE *tmp;

  /////////////////////////
  size_t oldSizes[1] = { N };
  size_t newSizes[1];
  getNewSizes(oldSizes, NULL, newSizes, NULL, "gesummv_kernel", 1);
  N = newSizes[0];
  /////////////////////////

  A = (DATA_TYPE *)malloc(N * N * sizeof(DATA_TYPE));
  B = (DATA_TYPE *)malloc(N * N * sizeof(DATA_TYPE));
  x = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  y = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  y_outputFromGpu = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  tmp = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));

  init(A, x);
  read_cl_file();
  cl_initialization(device_id, clGPUContext, clCommandQue);
  cl_mem_init(A, B, x, y, tmp);
  cl_load_prog();

  cl_launch_kernel();

  errcode = clEnqueueReadBuffer(clCommandQue, y_mem_obj, CL_TRUE, 0,
                                N * sizeof(DATA_TYPE), y_outputFromGpu, 0, NULL,
                                NULL);
  if (errcode != CL_SUCCESS)
    printf("Error in reading GPU mem\n");

  gesummv(A, B, x, y, tmp, y_outputFromGpu);
  cl_clean_up();

  free(A);
  free(B);
  free(x);
  free(y);
  free(y_outputFromGpu);
  free(tmp);

  return 0;
}
