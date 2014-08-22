/**
 * atax.cl: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#if defined(cl_khr_fp64) // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64) // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

typedef float DATA_TYPE;

// Traversal by rows.
__kernel void atax_kernel1(__global DATA_TYPE *A, __global DATA_TYPE *x,
                           __global DATA_TYPE *tmp, int rows, int columns) {

  int row = get_global_id(0);

  if (row < rows) {
    int column;
    for (column = 0; column < columns; column++) {
      tmp[row] += A[row * columns + column] * x[column];
    }
  }
}

// Traversal by columns.
__kernel void atax_kernel2(__global DATA_TYPE *A, __global DATA_TYPE *y,
                           __global DATA_TYPE *tmp, int rows, int columns) {

  int column = get_global_id(0);

  if (column < columns) {
    int row;
    for (row = 0; row < rows; row++) {
      y[column] += A[row * column + column] * tmp[row];
    }
  }
}
