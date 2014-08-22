/**
 * syrk.c: This file is part of the PolyBench/GPU 1.0 test suite.
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
#define N_DEFAULT 128 
#define M_DEFAULT 128 

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 32
#define DIM_LOCAL_WORK_GROUP_Y 8

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

char str_temp[1024];

DATA_TYPE acc;

DATA_TYPE alpha = 123;
DATA_TYPE beta = 14512;

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
cl_mem c_mem_obj;

FILE *fp;
char *source_str;
size_t source_size;

int N = N_DEFAULT;
int M = M_DEFAULT;

void read_cl_file()
{
	// Load the kernel source code into the array source_str
	fp = fopen(KERNEL_PATH "/syrk.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}


void init_arrays(DATA_TYPE* A, DATA_TYPE* C)
{
	int i, j;
	
	for (i = 0; i < N; i++)
    	{
		for (j = 0; j < M; j++)
		{
			A[i*M + j] = random<DATA_TYPE>();
		}
		
		for (j = 0; j < N; j++)
		{
			C[i*M + j] = random<DATA_TYPE>();
		}
	}
}

void cl_mem_init(DATA_TYPE* A, DATA_TYPE* C)
{
	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N * M, NULL, &errcode);
	c_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N * M, NULL, &errcode);
	
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N * M, A, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, c_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N * M, C, 0, NULL, NULL);
	if(errcode != CL_SUCCESS)printf("Error in writing buffers\n");
}


void cl_load_prog()
{
	// Create a program from the kernel source
	clProgram = clCreateProgramWithSource(clGPUContext, 1, (const char **)&source_str, (const size_t *)&source_size, &errcode);

	if(errcode != CL_SUCCESS) printf("Error in creating program\n");

	// Build the program
	errcode = clBuildProgram(clProgram, 1, &device_id, NULL, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in building program\n");
		
	// Create the OpenCL kernel
	clKernel1 = clCreateKernel(clProgram, "syrk_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel1\n");
	clFinish(clCommandQue);
}


void cl_launch_kernel()
{
	int m = M;
	int n = N;

  size_t oldLocalWorkSize[2], globalWorkSize[2];
  oldLocalWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
  oldLocalWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
  globalWorkSize[0] = N;
  globalWorkSize[1] = M;

  ///////////////////////////////////////////////
  size_t localWorkSize[2];
  getNewSizes(NULL, oldLocalWorkSize, NULL, localWorkSize,
              "syrk_kernel", 2);
  ///////////////////////////////////////////////


	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 1, sizeof(cl_mem), (void *)&c_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 2, sizeof(DATA_TYPE), (void *)&alpha);
	errcode |= clSetKernelArg(clKernel1, 3, sizeof(DATA_TYPE), (void *)&beta);
	errcode |= clSetKernelArg(clKernel1, 4, sizeof(int), (void *)&m);
	errcode |= clSetKernelArg(clKernel1, 5, sizeof(int), (void *)&n);

	if(errcode != CL_SUCCESS) printf("Error in seting arguments1\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel1, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel1\n");
	clFinish(clCommandQue);
}


void cl_clean_up()
{
	// Clean up
	errcode = clFlush(clCommandQue);
	errcode = clFinish(clCommandQue);
	errcode = clReleaseKernel(clKernel1);
	errcode = clReleaseProgram(clProgram);
	errcode = clReleaseMemObject(a_mem_obj);
	errcode = clReleaseMemObject(c_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}

void syrk(DATA_TYPE *A, DATA_TYPE *C, DATA_TYPE *result) {
  int i, j, k;

  char *reps = getEnvString("OCL_REPETITIONS");
  int intReps = 1;
  if (reps != NULL) {
    intReps = atoi(reps);
  }

  for (i = 0; i < 128; i++) {
    for (j = 0; j < 128; j++) {
      for (int rep = 0; rep < intReps; ++rep) {
        C[i * M + j] *= beta;
        for (k = 0; k < M; k++) {
          C[i * N + j] += alpha * A[i * M + k] * A[j * M + k];
        }
      }

      assert(fabs(C[i * N + j] - result[i * N + j])  <
                 0.001 &&
             "Error!");
    }
  }

  std::cout << "Ok!\n";
}

int main(void) 
{
	DATA_TYPE* A;
	DATA_TYPE* C;
	DATA_TYPE* C_outputFromGpu;

  /////////////////////////
  size_t oldSizes[2] = { N, M };
  size_t newSizes[2];
  getNewSizes(oldSizes, NULL, newSizes, NULL, "syrk_kernel", 2);
  N = newSizes[0];
  M = newSizes[1];
  /////////////////////////

	A = (DATA_TYPE*)malloc(N*M*sizeof(DATA_TYPE));
	C = (DATA_TYPE*)malloc(N*M*sizeof(DATA_TYPE));
	C_outputFromGpu = (DATA_TYPE*)malloc(N*M*sizeof(DATA_TYPE));

	init_arrays(A, C);
	read_cl_file();
  cl_initialization(device_id, clGPUContext, clCommandQue);
	cl_mem_init(A, C);
	cl_load_prog();

	cl_launch_kernel();

	errcode = clEnqueueReadBuffer(clCommandQue, c_mem_obj, CL_TRUE, 0, M * N * sizeof(DATA_TYPE), C_outputFromGpu, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");  

	syrk(A, C, C_outputFromGpu);
	cl_clean_up();
	
	free(A);
	free(C);
	free(C_outputFromGpu);

	return 0;
}

