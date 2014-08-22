/**
 * bicg.c: This file is part of the PolyBench/GPU 1.0 test suite.
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

/* Problem size. */
#define NX_DEFAULT 4096

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 256

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
cl_mem r_mem_obj;
cl_mem p_mem_obj;
cl_mem q_mem_obj;
cl_mem s_mem_obj;

FILE *fp;
char *source_str;
size_t source_size;

size_t NX = NX_DEFAULT;

void read_cl_file() {
  // Load the kernel source code into the array source_str
  fp = fopen(KERNEL_PATH "/bicg.cl", "r");
  if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
  }
  source_str = (char *)malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);
}

void init_array(DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *r) {
  int i, j;

  for (i = 0; i < NX; i++) {
    r[i] = i * M_PI;

    for (j = 0; j < NX; j++) {
      A[i * NX + j] = random<DATA_TYPE>();
    }
  }

  for (i = 0; i < NX; i++) {
    p[i] = i * M_PI;
  }
}

void cl_mem_init(DATA_TYPE *A, DATA_TYPE *r, DATA_TYPE *s, DATA_TYPE *p,
                 DATA_TYPE *q) {
  a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE,
                             sizeof(DATA_TYPE) * NX * NX, NULL, &errcode);
  r_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE,
                             sizeof(DATA_TYPE) * NX, NULL, &errcode);
  s_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE,
                             sizeof(DATA_TYPE) * NX, NULL, &errcode);
  p_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE,
                             sizeof(DATA_TYPE) * NX, NULL, &errcode);
  q_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE,
                             sizeof(DATA_TYPE) * NX, NULL, &errcode);

  if (errcode != CL_SUCCESS)
    printf("Error in creating buffers\n");

  errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0,
                                 sizeof(DATA_TYPE) * NX * NX, A, 0, NULL, NULL);
  errcode = clEnqueueWriteBuffer(clCommandQue, r_mem_obj, CL_TRUE, 0,
                                 sizeof(DATA_TYPE) * NX, r, 0, NULL, NULL);
  errcode = clEnqueueWriteBuffer(clCommandQue, s_mem_obj, CL_TRUE, 0,
                                 sizeof(DATA_TYPE) * NX, s, 0, NULL, NULL);
  errcode = clEnqueueWriteBuffer(clCommandQue, p_mem_obj, CL_TRUE, 0,
                                 sizeof(DATA_TYPE) * NX, p, 0, NULL, NULL);
  errcode = clEnqueueWriteBuffer(clCommandQue, q_mem_obj, CL_TRUE, 0,
                                 sizeof(DATA_TYPE) * NX, q, 0, NULL, NULL);
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
  clKernel1 = clCreateKernel(clProgram, "bicgKernel1", &errcode);
  if (errcode != CL_SUCCESS)
    printf("Error in creating kernel\n");

  //	// Create the 2nd OpenCL kernel
  //	clKernel2 = clCreateKernel(clProgram, "bicgKernel2", &errcode);
  //	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");

  clFinish(clCommandQue);
}

void cl_launch_kernel() {
  int nx = NX;
  int ny = NX;

  size_t oldLocalWorkSize[1], globalWorkSize[1];
  oldLocalWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
  globalWorkSize[0] = NX;

  ///////////////////////////////////////////////
  size_t localWorkSize[1];
  getNewSizes(NULL, oldLocalWorkSize, NULL, localWorkSize, "bicgKernel1", 1);
  ///////////////////////////////////////////////

  // Set the arguments of the kernel
  errcode = clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&a_mem_obj);
  errcode |= clSetKernelArg(clKernel1, 1, sizeof(cl_mem), (void *)&p_mem_obj);
  errcode |= clSetKernelArg(clKernel1, 2, sizeof(cl_mem), (void *)&q_mem_obj);
  errcode |= clSetKernelArg(clKernel1, 3, sizeof(int), &nx);
  errcode |= clSetKernelArg(clKernel1, 4, sizeof(int), &ny);
  if (errcode != CL_SUCCESS)
    printf("Error in seting arguments\n");

  // Execute the 1st OpenCL kernel
  errcode =
      clEnqueueNDRangeKernel(clCommandQue, clKernel1, 1, NULL, globalWorkSize,
                             localWorkSize, 0, NULL, NULL);
  if (errcode != CL_SUCCESS)
    printf("Error in launching kernel\n");
  clFinish(clCommandQue);

  //	globalWorkSize[0] = (size_t)ceil(((float)NX) /
  //((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
  //	globalWorkSize[1] = 1;
  //
  //	// Set the arguments of the kernel
  //	errcode =  clSetKernelArg(clKernel2, 0, sizeof(cl_mem), (void
  //*)&a_mem_obj);
  //	errcode |= clSetKernelArg(clKernel2, 1, sizeof(cl_mem), (void
  //*)&r_mem_obj);
  //	errcode |= clSetKernelArg(clKernel2, 2, sizeof(cl_mem), (void
  //*)&s_mem_obj);
  //	errcode |= clSetKernelArg(clKernel2, 3, sizeof(int), &nx);
  //        errcode |= clSetKernelArg(clKernel2, 4, sizeof(int), &ny);
  //	if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");
  //
  //	// Execute the 2nd OpenCL kernel
  //	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel2, 1, NULL,
  //globalWorkSize, localWorkSize, 0, NULL, NULL);
  //	if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
  //	clFinish(clCommandQue);
}

void cl_clean_up() {
  // Clean up
  errcode = clFlush(clCommandQue);
  errcode = clFinish(clCommandQue);
  errcode = clReleaseKernel(clKernel1);
  errcode = clReleaseKernel(clKernel2);
  errcode = clReleaseProgram(clProgram);
  errcode = clReleaseMemObject(a_mem_obj);
  errcode = clReleaseMemObject(p_mem_obj);
  errcode = clReleaseMemObject(q_mem_obj);
  errcode = clReleaseMemObject(r_mem_obj);
  errcode = clReleaseMemObject(s_mem_obj);
  errcode = clReleaseCommandQueue(clCommandQue);
  errcode = clReleaseContext(clGPUContext);
  if (errcode != CL_SUCCESS)
    printf("Error in cleanup\n");
}

void bicg_cpu(DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *q, DATA_TYPE *result) {
  int i, j;

  char *reps = getEnvString("OCL_REPETITIONS");
  int intReps = 1;
  if (reps != NULL) {
    intReps = atoi(reps);
  }

  for (i = 0; i < NX; i++) {
    q[i] = 0.0;

    for (int rep = 0; rep < intReps; ++rep) {
      for (j = 0; j < NX; j++) {
        q[i] += A[i * NX + j] * p[j];
      }
    }

    assert(fabs(q[i] - result[i]) / result[i] < 0.05 && "Error in the computation");
  }

  std::cout << "Ok!\n";
}

int main(void) {
  DATA_TYPE *A;
  DATA_TYPE *r;
  DATA_TYPE *s;
  DATA_TYPE *p;
  DATA_TYPE *q;
  DATA_TYPE *s_outputFromGpu;
  DATA_TYPE *q_outputFromGpu;

  /////////////////////////
  size_t oldSizes[1] = { NX };
  size_t newSizes[1];
  getNewSizes(oldSizes, NULL, newSizes, NULL, "bicgKernel1", 1);
  NX = newSizes[0];
  /////////////////////////

  A = (DATA_TYPE *)malloc(NX * NX * sizeof(DATA_TYPE));
  r = (DATA_TYPE *)malloc(NX * sizeof(DATA_TYPE));
  s = (DATA_TYPE *)malloc(NX * sizeof(DATA_TYPE));
  p = (DATA_TYPE *)malloc(NX * sizeof(DATA_TYPE));
  q = (DATA_TYPE *)malloc(NX * sizeof(DATA_TYPE));
  s_outputFromGpu = (DATA_TYPE *)malloc(NX * sizeof(DATA_TYPE));
  q_outputFromGpu = (DATA_TYPE *)malloc(NX * sizeof(DATA_TYPE));

  init_array(A, p, r);
  read_cl_file();
  cl_initialization(device_id, clGPUContext, clCommandQue);
  cl_mem_init(A, r, s, p, q);
  cl_load_prog();

  cl_launch_kernel();

//  errcode = clEnqueueReadBuffer(clCommandQue, s_mem_obj, CL_TRUE, 0,
//                                NX * sizeof(DATA_TYPE), s_outputFromGpu, 0,
//                                NULL, NULL);
  errcode = clEnqueueReadBuffer(clCommandQue, q_mem_obj, CL_TRUE, 0,
                                NX * sizeof(DATA_TYPE), q_outputFromGpu, 0,
                                NULL, NULL);
  if (errcode != CL_SUCCESS)
    printf("Error in reading GPU mem\n");

  bicg_cpu(A, p, q, q_outputFromGpu);
  cl_clean_up();

  free(A);
  free(r);
  free(s);
  free(p);
  free(q);
  free(s_outputFromGpu);
  free(q_outputFromGpu);

  return 0;
}
