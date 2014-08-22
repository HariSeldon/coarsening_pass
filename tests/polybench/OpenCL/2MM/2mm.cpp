/**
 * 2mm.c: This file is part of the PolyBench/GPU 1.0 test suite.
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
#define NI_DEFAULT 2048
#define NJ_DEFAULT 2048
#define NK_DEFAULT 2048
#define NL_DEFAULT 2048

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
cl_kernel clKernel1;
cl_kernel clKernel2;
cl_command_queue clCommandQue;
cl_program clProgram;
cl_mem a_mem_obj;
cl_mem b_mem_obj;
cl_mem c_mem_obj;
cl_mem d_mem_obj;
cl_mem e_mem_obj;

size_t NJ = NJ_DEFAULT;
size_t NK = NK_DEFAULT;
size_t NI = NI_DEFAULT;
size_t NL = NL_DEFAULT;

FILE *fp;
char *source_str;
size_t source_size;

void read_cl_file() {
  // Load the kernel source code into the array source_str
  fp = fopen(KERNEL_PATH "/2mm.cl", "r");
  if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
  }
  source_str = (char *)malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);
}

void init_array(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D) {
  int i, j;

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NK; j++) {
      A[i * NI + j] = random<DATA_TYPE>();
    }
  }

  for (i = 0; i < NK; i++) {
    for (j = 0; j < NJ; j++) {
      B[i * NK + j] = random<DATA_TYPE>();
    }
  }
}

void cl_mem_init(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D,
                 DATA_TYPE *E) {
  a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_ONLY,
                             sizeof(DATA_TYPE) * NI * NK, NULL, &errcode);
  b_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_ONLY,
                             sizeof(DATA_TYPE) * NK * NJ, NULL, &errcode);
  c_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE,
                             sizeof(DATA_TYPE) * NI * NJ, NULL, &errcode);
  d_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE,
                             sizeof(DATA_TYPE) * NJ * NL, NULL, &errcode);
  e_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE,
                             sizeof(DATA_TYPE) * NI * NL, NULL, &errcode);

  if (errcode != CL_SUCCESS)
    printf("Error in creating buffers\n");

  errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0,
                                 sizeof(DATA_TYPE) * NI * NK, A, 0, NULL, NULL);
  errcode = clEnqueueWriteBuffer(clCommandQue, b_mem_obj, CL_TRUE, 0,
                                 sizeof(DATA_TYPE) * NK * NJ, B, 0, NULL, NULL);
  errcode = clEnqueueWriteBuffer(clCommandQue, c_mem_obj, CL_TRUE, 0,
                                 sizeof(DATA_TYPE) * NI * NJ, C, 0, NULL, NULL);
  errcode = clEnqueueWriteBuffer(clCommandQue, d_mem_obj, CL_TRUE, 0,
                                 sizeof(DATA_TYPE) * NJ * NL, D, 0, NULL, NULL);
  errcode = clEnqueueWriteBuffer(clCommandQue, e_mem_obj, CL_TRUE, 0,
                                 sizeof(DATA_TYPE) * NI * NL, E, 0, NULL, NULL);
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
  clKernel1 = clCreateKernel(clProgram, "mm2_kernel1", &errcode);
  if (errcode != CL_SUCCESS)
    printf("Error in creating kernel\n");

//  clKernel2 = clCreateKernel(clProgram, "mm2_kernel2", &errcode);
//  if (errcode != CL_SUCCESS)
//    printf("Error in creating kernel\n");
  clFinish(clCommandQue);
}

void cl_launch_kernel() {
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;

  size_t oldLocalWorkSize[2], globalWorkSize[2];
  oldLocalWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
  oldLocalWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
  globalWorkSize[0] = NI;
  globalWorkSize[1] = NL;

  ///////////////////////////////////////////////
  size_t localWorkSize[2];
  getNewSizes(NULL, oldLocalWorkSize, NULL, localWorkSize,
              "mm2_kernel1", 2);
  ///////////////////////////////////////////////

  // Set the arguments of the kernel
  errcode = clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&a_mem_obj);
  errcode |= clSetKernelArg(clKernel1, 1, sizeof(cl_mem), (void *)&b_mem_obj);
  errcode |= clSetKernelArg(clKernel1, 2, sizeof(cl_mem), (void *)&c_mem_obj);
  errcode |= clSetKernelArg(clKernel1, 3, sizeof(int), (void *)&ni);
  errcode |= clSetKernelArg(clKernel1, 4, sizeof(int), (void *)&nk);
  errcode |= clSetKernelArg(clKernel1, 5, sizeof(int), (void *)&nj);
  if (errcode != CL_SUCCESS)
    printf("Error in seting arguments\n");
  // Execute the OpenCL kernel
  errcode =
      clEnqueueNDRangeKernel(clCommandQue, clKernel1, 2, NULL, globalWorkSize,
                             localWorkSize, 0, NULL, NULL);
  if (errcode != CL_SUCCESS)
    printf("Error in launching kernel\n");
//  clEnqueueBarrier(clCommandQue);
//
//  globalWorkSize[0] = NI;
//  globalWorkSize[1] = NL;
//
//  errcode = clSetKernelArg(clKernel2, 0, sizeof(cl_mem), (void *)&c_mem_obj);
//  errcode |= clSetKernelArg(clKernel2, 1, sizeof(cl_mem), (void *)&d_mem_obj);
//  errcode |= clSetKernelArg(clKernel2, 2, sizeof(cl_mem), (void *)&e_mem_obj);
//  errcode |= clSetKernelArg(clKernel2, 3, sizeof(int), (void *)&ni);
//  errcode |= clSetKernelArg(clKernel2, 4, sizeof(int), (void *)&nj);
//  errcode |= clSetKernelArg(clKernel2, 5, sizeof(int), (void *)&nl);
//  if (errcode != CL_SUCCESS)
//    printf("Error in seting arguments\n");
//
//  // Execute the OpenCL kernel
//  errcode =
//      clEnqueueNDRangeKernel(clCommandQue, clKernel2, 2, NULL, globalWorkSize,
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
  errcode = clReleaseMemObject(b_mem_obj);
  errcode = clReleaseMemObject(c_mem_obj);
  errcode = clReleaseMemObject(d_mem_obj);
  errcode = clReleaseMemObject(e_mem_obj);
  errcode = clReleaseCommandQueue(clCommandQue);
  errcode = clReleaseContext(clGPUContext);
  if (errcode != CL_SUCCESS)
    printf("Error in cleanup\n");
}

void mm2_cpu(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C) {
  int i, j, k;

  char *reps = getEnvString("OCL_REPETITIONS");
  int intReps = 1;
  if(reps != NULL) {
    intReps = atoi(reps);
  }

  for (i = 0; i < NI; i++) {
    for (j = 0; j < 16; j++) {
      DATA_TYPE tmp = 0;
      
      for (int rep = 0; rep < intReps; ++rep) {
      for (k = 0; k < NK; ++k) {
        tmp += A[i * NK + k] * B[k * NJ + j];
      }
      }
      DATA_TYPE diff = fabs(C[i * NJ + j] - tmp); 
      assert(diff < 1 && "Error!"); 
    }
  }

  std::cout << "Ok!\n";
}

int main(void) {
  DATA_TYPE *C;
  DATA_TYPE *A;
  DATA_TYPE *B;
  DATA_TYPE *D;
  DATA_TYPE *E;
  DATA_TYPE *E_outputFromGpu;

  /////////////////////////
  size_t oldSizes[2] = { NI, NL };
  size_t newSizes[2];
  getNewSizes(oldSizes, NULL, newSizes, NULL, "mm2_kernel1", 2);
//  size_t tmpSizes[2] = {newSizes[0], newSizes[1]};
//  getNewSizes(tmpSizes, NULL, newSizes, NULL, "mm2_kernel2", 2);
  NI = newSizes[0];
  NL = newSizes[1];
  NJ = NI;
  NK = NI;
  /////////////////////////

  C = (DATA_TYPE *)malloc(NI * NJ * sizeof(DATA_TYPE));
  A = (DATA_TYPE *)malloc(NI * NK * sizeof(DATA_TYPE));
  B = (DATA_TYPE *)malloc(NK * NJ * sizeof(DATA_TYPE));
  D = (DATA_TYPE *)malloc(NJ * NL * sizeof(DATA_TYPE));
  E = (DATA_TYPE *)malloc(NI * NL * sizeof(DATA_TYPE));
  E_outputFromGpu = (DATA_TYPE *)malloc(NI * NL * sizeof(DATA_TYPE));

  int i;
  init_array(A, B, C, D);
  read_cl_file();
  cl_initialization(device_id, clGPUContext, clCommandQue);
  cl_mem_init(A, B, C, D, E);
  cl_load_prog();

  cl_launch_kernel();

  errcode = clEnqueueReadBuffer(clCommandQue, c_mem_obj, CL_TRUE, 0,
                                sizeof(DATA_TYPE) * NI * NL, E_outputFromGpu, 0,
                                NULL, NULL);
  if (errcode != CL_SUCCESS)
    printf("Error in reading GPU mem\n");

  mm2_cpu(A, B, E_outputFromGpu);
  cl_clean_up();

  free(C);
  free(A);
  free(B);
  free(D);
  free(E);
  free(E_outputFromGpu);

  return 0;
}
