/**
 * gramschmidt.c: This file is part of the PolyBench/GPU 1.0 test suite.
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
#define M_DEFAULT 1024 
#define N_DEFAULT 1024 

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

typedef double DATA_TYPE;

char str_temp[1024];
 
int M = M_DEFAULT;
int N = N_DEFAULT;

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
cl_mem r_mem_obj;
cl_mem q_mem_obj;

FILE *fp;
char *source_str;
size_t source_size;

void compareResults(DATA_TYPE* A, DATA_TYPE* A_outputFromGpu)
{
	int i, j, fail;
	fail = 0;

	for (i=0; i < M; i++) 
	{
		for (j=0; j < N; j++) 
		{
			if (percentDiff(A[i*N + j], A_outputFromGpu[i*N + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{				
				fail++;
				printf("i: %d j: %d \n1: %f\n 2: %f\n", i, j, A[i*N + j], A_outputFromGpu[i*N + j]);
			}
		}
	}

  assert(fail == 0 && "CPU - GPU Computation does not match!");
}


void read_cl_file()
{
	// Load the kernel source code into the array source_str
	fp = fopen(KERNEL_PATH "/gramschmidt.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}


void init_array(DATA_TYPE* A)
{
	int i, j;

	for (i = 0; i < M; i++)
	{
		for (j = 0; j < N; j++)
		{
			A[i*N + j] = ((DATA_TYPE) (i+1)*(j+1)) / (M+1);
		}
	}
}

void cl_mem_init(DATA_TYPE* A)
{
	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * M * N, NULL, &errcode);
	r_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * M * N, NULL, &errcode);
	q_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * M * N, NULL, &errcode);
	
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * M * N, A, 0, NULL, NULL);
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
	clKernel1 = clCreateKernel(clProgram, "gramschmidt_kernel1", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel1\n");

	clKernel2 = clCreateKernel(clProgram, "gramschmidt_kernel2", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel2\n");

	clKernel3 = clCreateKernel(clProgram, "gramschmidt_kernel3", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel3\n");
	clFinish(clCommandQue);
}


void cl_launch_kernel()
{
	int m = M;
	int n = N;

  size_t oldLocalWorkSize[2], globalWorkSizeKernel1[2], localWorkSize[2]; 
  size_t globalWorkSizeKernel2[2], globalWorkSizeKernel3[2];

  oldLocalWorkSize[0] = DIM_THREAD_BLOCK_X;
	oldLocalWorkSize[1] = DIM_THREAD_BLOCK_Y;
	globalWorkSizeKernel1[0] = DIM_THREAD_BLOCK_X;
	globalWorkSizeKernel1[1] = DIM_THREAD_BLOCK_Y;
	globalWorkSizeKernel2[0] = N;
	globalWorkSizeKernel2[1] = 1;
	globalWorkSizeKernel3[0] = N;
	globalWorkSizeKernel3[1] = 1;

  ///////////////////////////////////////////////
  // Kernel 2.
  getNewSizes(NULL, oldLocalWorkSize, NULL, localWorkSize, "gramschmidt_kernel2", 2);
  // Kernel 3.
  getNewSizes(NULL, localWorkSize, NULL, localWorkSize, "gramschmidt_kernel3", 2);

  ///////////////////////////////////////////////

	int k;
	for (k = 0; k < 1; k++)
	{
		// Set the arguments of the kernel
		errcode =  clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&a_mem_obj);
		errcode =  clSetKernelArg(clKernel1, 1, sizeof(cl_mem), (void *)&r_mem_obj);
		errcode |= clSetKernelArg(clKernel1, 2, sizeof(cl_mem), (void *)&q_mem_obj);
		errcode |= clSetKernelArg(clKernel1, 3, sizeof(int), (void *)&k);
		errcode |= clSetKernelArg(clKernel1, 4, sizeof(int), (void *)&m);
		errcode |= clSetKernelArg(clKernel1, 5, sizeof(int), (void *)&n);
	
		if(errcode != CL_SUCCESS) printf("Error in seting arguments1\n");
	
		// Execute the OpenCL kernel
		errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel1, 1, NULL, globalWorkSizeKernel1, localWorkSize, 0, NULL, NULL);
		if(errcode != CL_SUCCESS) printf("Error in launching kernel1\n");
		clEnqueueBarrier(clCommandQue);


		errcode =  clSetKernelArg(clKernel2, 0, sizeof(cl_mem), (void *)&a_mem_obj);
		errcode =  clSetKernelArg(clKernel2, 1, sizeof(cl_mem), (void *)&r_mem_obj);
		errcode |= clSetKernelArg(clKernel2, 2, sizeof(cl_mem), (void *)&q_mem_obj);
		errcode |= clSetKernelArg(clKernel2, 3, sizeof(int), (void *)&k);
		errcode |= clSetKernelArg(clKernel2, 4, sizeof(int), (void *)&m);
		errcode |= clSetKernelArg(clKernel2, 5, sizeof(int), (void *)&n);
	
		if(errcode != CL_SUCCESS) printf("Error in seting arguments1\n");
	
		// Execute the OpenCL kernel
		errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel2, 1, NULL, globalWorkSizeKernel2, localWorkSize, 0, NULL, NULL);
		if(errcode != CL_SUCCESS) printf("Error in launching kernel2\n");
		clEnqueueBarrier(clCommandQue);


		errcode =  clSetKernelArg(clKernel3, 0, sizeof(cl_mem), (void *)&a_mem_obj);
		errcode =  clSetKernelArg(clKernel3, 1, sizeof(cl_mem), (void *)&r_mem_obj);
		errcode |= clSetKernelArg(clKernel3, 2, sizeof(cl_mem), (void *)&q_mem_obj);
		errcode |= clSetKernelArg(clKernel3, 3, sizeof(int), (void *)&k);
		errcode |= clSetKernelArg(clKernel3, 4, sizeof(int), (void *)&m);
		errcode |= clSetKernelArg(clKernel3, 5, sizeof(int), (void *)&n);
	
		if(errcode != CL_SUCCESS) printf("Error in seting arguments1\n");
	
		// Execute the OpenCL kernel
		errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel3, 1, NULL, globalWorkSizeKernel3, localWorkSize, 0, NULL, NULL);
		if(errcode != CL_SUCCESS) printf("Error in launching kernel3\n");
		clEnqueueBarrier(clCommandQue);

	}
	clFinish(clCommandQue);

}


void cl_clean_up()
{
	// Clean up
	errcode = clFlush(clCommandQue);
	errcode = clFinish(clCommandQue);
	errcode = clReleaseKernel(clKernel1);
	errcode = clReleaseKernel(clKernel2);
	errcode = clReleaseKernel(clKernel3);
	errcode = clReleaseProgram(clProgram);
	errcode = clReleaseMemObject(a_mem_obj);
	errcode = clReleaseMemObject(r_mem_obj);
	errcode = clReleaseMemObject(q_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}


void gramschmidt(DATA_TYPE* A, DATA_TYPE* R, DATA_TYPE* Q)
{
	int i,j,k;
	DATA_TYPE nrm;
	for (k = 0; k < N; k++)
	{
		nrm = 0;
		for (i = 0; i < M; i++)
		{
			nrm += A[i*N + k] * A[i*N + k];
		}
		
		R[k*N + k] = sqrt(nrm);
		for (i = 0; i < M; i++)
		{
			Q[i*N + k] = A[i*N + k] / R[k*N + k];
		}
		
		for (j = k + 1; j < N; j++)
		{
			R[k*N + j] = 0;
			for (i = 0; i < M; i++)
			{
				R[k*N + j] += Q[i*N + k] * A[i*N + j];
			}
			for (i = 0; i < M; i++)
			{
				A[i*N + j] = A[i*N + j] - Q[i*N + k] * R[k*N + j];
			}
		}
	}
}


int main(void) 
{
	DATA_TYPE* A;
	DATA_TYPE* A_outputFromGpu;
	DATA_TYPE* R;
	DATA_TYPE* Q;
	
  /////////////////////////
  // Kernel 1.
  size_t oldSizes[2] = { M, N };
  size_t newSizes[2];
  getNewSizes(oldSizes, NULL, newSizes, NULL, "gramschmidt_kernel1", 2);
  M = newSizes[0];
  N = newSizes[1];

  // Kernel 2.
  getNewSizes(newSizes, NULL, newSizes, NULL, "gramschmidt_kernel2", 2);
  M = newSizes[0];
  N = newSizes[1];

  // Kernel 3.
  getNewSizes(newSizes, NULL, newSizes, NULL, "gramschmidt_kernel3", 2);
  M = newSizes[0];
  N = newSizes[1];
  /////////////////////////

	A = (DATA_TYPE*)malloc(M*N*sizeof(DATA_TYPE));
	A_outputFromGpu = (DATA_TYPE*)malloc(M*N*sizeof(DATA_TYPE));
	R = (DATA_TYPE*)malloc(M*N*sizeof(DATA_TYPE));  
	Q = (DATA_TYPE*)malloc(M*N*sizeof(DATA_TYPE));  

	init_array(A);
	read_cl_file();
  cl_initialization(device_id, clGPUContext, clCommandQue);
	cl_mem_init(A);
	cl_load_prog();

	cl_launch_kernel();

	errcode = clEnqueueReadBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, M*N*sizeof(DATA_TYPE), A_outputFromGpu, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");   

//	gramschmidt(A, R, Q);
//	compareResults(A, A_outputFromGpu);
	cl_clean_up();

	free(A);
	free(A_outputFromGpu);
	free(R);
	free(Q);  

	return 0;
}

