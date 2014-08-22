/**
 * mvt.c: This file is part of the PolyBench/GPU 1.0 test suite.
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
typedef double DATA_TYPE;

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
cl_mem x1_mem_obj;
cl_mem x2_mem_obj;
cl_mem y1_mem_obj;
cl_mem y2_mem_obj;

FILE *fp;
char *source_str;
size_t source_size;
char str_temp[1024];

size_t N = N_DEFAULT;

void read_cl_file() {
  // Load the kernel source code into the array source_str
  fp = fopen(KERNEL_PATH "/mvt.cl", "r");
  if (!fp) {
    fprintf(stdout, "Failed to load kernel.\n");
    exit(1);
  }
  source_str = (char *)malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);
}

void init_arrays(DATA_TYPE *a, DATA_TYPE *x1, DATA_TYPE *x2, DATA_TYPE *y_1,
                 DATA_TYPE *y_2) {
  int i, j;

  for (i = 0; i < N; i++) {
    x1[i] = 0.0;
    x2[i] = 0.0;
    y_1[i] = 0.0;
    y_2[i] = 0.0;

    for (j = 0; j < N; j++) {
      a[i * N + j] = random<DATA_TYPE>();
    }
  }
}

void cl_mem_init(DATA_TYPE *a, DATA_TYPE *x1, DATA_TYPE *x2, DATA_TYPE *y_1,
                 DATA_TYPE *y_2) {
  a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE,
                             sizeof(DATA_TYPE) * N * N, NULL, &errcode);
  x1_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE,
                              sizeof(DATA_TYPE) * N, NULL, &errcode);
  x2_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE,
                              sizeof(DATA_TYPE) * N, NULL, &errcode);
  y1_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE,
                              sizeof(DATA_TYPE) * N, NULL, &errcode);
  y2_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE,
                              sizeof(DATA_TYPE) * N, NULL, &errcode);

  if (errcode != CL_SUCCESS)
    printf("Error in creating buffers\n");

  errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0,
                                 sizeof(DATA_TYPE) * N * N, a, 0, NULL, NULL);
  errcode = clEnqueueWriteBuffer(clCommandQue, x1_mem_obj, CL_TRUE, 0,
                                 sizeof(DATA_TYPE) * N, x1, 0, NULL, NULL);
  errcode = clEnqueueWriteBuffer(clCommandQue, x2_mem_obj, CL_TRUE, 0,
                                 sizeof(DATA_TYPE) * N, x2, 0, NULL, NULL);
  errcode = clEnqueueWriteBuffer(clCommandQue, y1_mem_obj, CL_TRUE, 0,
                                 sizeof(DATA_TYPE) * N, y_1, 0, NULL, NULL);
  errcode = clEnqueueWriteBuffer(clCommandQue, y2_mem_obj, CL_TRUE, 0,
                                 sizeof(DATA_TYPE) * N, y_2, 0, NULL, NULL);

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
    printf("Error in building program %d\n", errcode);

  // Create the 1st OpenCL kernel
  clKernel1 = clCreateKernel(clProgram, "mvt_kernel1", &errcode);
  //  // Create the 2nd OpenCL kernel
  //  clKernel2 = clCreateKernel(clProgram, "mvt_kernel2", &errcode);
  if (errcode != CL_SUCCESS)
    printf("Error in creating kernel\n");
  clFinish(clCommandQue);
}

void cl_launch_kernel() {
  int n = N;

  size_t oldLocalWorkSize[1], globalWorkSize[1];
  oldLocalWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
  globalWorkSize[0] = N;

  ///////////////////////////////////////////////
  size_t localWorkSize[1];
  getNewSizes(NULL, oldLocalWorkSize, NULL, localWorkSize, "mvt_kernel1", 1);
  ///////////////////////////////////////////////

  // Set the arguments of the kernel
  errcode = clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&a_mem_obj);
  errcode |= clSetKernelArg(clKernel1, 1, sizeof(cl_mem), (void *)&x1_mem_obj);
  errcode |= clSetKernelArg(clKernel1, 2, sizeof(cl_mem), (void *)&y1_mem_obj);
  errcode |= clSetKernelArg(clKernel1, 3, sizeof(int), (void *)&n);
  if (errcode != CL_SUCCESS)
    printf("Error in seting arguments\n");

  // Execute the OpenCL kernel
  errcode =
      clEnqueueNDRangeKernel(clCommandQue, clKernel1, 1, NULL, globalWorkSize,
                             localWorkSize, 0, NULL, NULL);
  if (errcode != CL_SUCCESS)
    printf("Error in launching kernel\n");

  //  // Set the arguments of the kernel
  //  errcode = clSetKernelArg(clKernel2, 0, sizeof(cl_mem), (void
  // *)&a_mem_obj);
  //  errcode |= clSetKernelArg(clKernel2, 1, sizeof(cl_mem), (void
  // *)&x2_mem_obj);
  //  errcode |= clSetKernelArg(clKernel2, 2, sizeof(cl_mem), (void
  // *)&y2_mem_obj);
  //  errcode |= clSetKernelArg(clKernel2, 3, sizeof(int), (void *)&n);
  //  if (errcode != CL_SUCCESS)
  //    printf("Error in seting arguments\n");
  //
  //  // Execute the OpenCL kernel
  //  errcode =
  //      clEnqueueNDRangeKernel(clCommandQue, clKernel2, 1, NULL,
  // globalWorkSize,
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
  errcode = clReleaseMemObject(x1_mem_obj);
  errcode = clReleaseMemObject(x2_mem_obj);
  errcode = clReleaseMemObject(y1_mem_obj);
  errcode = clReleaseMemObject(y2_mem_obj);
  errcode = clReleaseCommandQueue(clCommandQue);
  errcode = clReleaseContext(clGPUContext);
  if (errcode != CL_SUCCESS)
    printf("Error in cleanup\n");
}

void runMvt(DATA_TYPE *a, DATA_TYPE *x1, DATA_TYPE *x2, DATA_TYPE *y1,
            DATA_TYPE *y2, DATA_TYPE *result) {
  int i, j, k, l;

  char *reps = getEnvString("OCL_REPETITIONS");
  int intReps = 1;
  if (reps != NULL) {
    intReps = atoi(reps);
  }

  for (i = 0; i < N; i++) {
    for (int rep = 0; rep < intReps; ++rep) {
      for (j = 0; j < N; j++) {
        x1[i] = x1[i] + a[i * N + j] * y1[j];
      }
    }

    assert(fabs(x1[i] - result[i]) < 0.1 && "Error!");
  }

  std::cout << "Ok!\n";
}

int main(void) {
  DATA_TYPE *a;
  DATA_TYPE *x1;
  DATA_TYPE *x2;
  DATA_TYPE *x1_outputFromGpu;
  DATA_TYPE *x2_outputFromGpu;
  DATA_TYPE *y_1;
  DATA_TYPE *y_2;

  /////////////////////////
  size_t oldSizes[1] = { N };
  size_t newSizes[1];
  getNewSizes(oldSizes, NULL, newSizes, NULL, "mvt_kernel1", 1);
  N = newSizes[0];
  /////////////////////////

  a = (DATA_TYPE *)malloc(N * N * sizeof(DATA_TYPE));
  x1 = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  x2 = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  x1_outputFromGpu = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  x2_outputFromGpu = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  y_1 = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  y_2 = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));

  init_arrays(a, x1, x2, y_1, y_2);
  read_cl_file();
  cl_initialization(device_id, clGPUContext, clCommandQue);
  cl_mem_init(a, x1, x2, y_1, y_2);
  cl_load_prog();

  cl_launch_kernel();

  errcode = clEnqueueReadBuffer(clCommandQue, x1_mem_obj, CL_TRUE, 0,
                                N * sizeof(DATA_TYPE), x1_outputFromGpu, 0,
                                NULL, NULL);
  if (errcode != CL_SUCCESS)
    printf("Error in reading GPU mem\n");

  runMvt(a, x1, x2, y_1, y_2, x1_outputFromGpu);
  cl_clean_up();

  free(a);
  free(x1);
  free(x2);
  free(x1_outputFromGpu);
  free(x2_outputFromGpu);
  free(y_1);
  free(y_2);

  return 0;
}

