__kernel void mm(const __global float* A,
                 const __global float* B,
                       __global float* C,
                 uint width, uint height) {
  uint row = get_global_id(1);
  uint column = get_global_id(0);

  float tmp = 0.0f;
  for(uint index = 0; index < width; ++index)
    tmp += A[row * width + index] * B[index * width + column];
  C[row * width + column] = tmp;
}
