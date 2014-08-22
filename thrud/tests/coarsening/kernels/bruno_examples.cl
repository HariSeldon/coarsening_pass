__kernel void vectorTest(__global float *input, __global float *output,
                         uint width, uint height) {

  uint row = get_global_id(1);
  uint column = get_global_id(0);

  float tmp = 0.0f;

  for (uint index = 0; index < width; ++index)
    tmp += input[width + index];

  output[row * width + column] = tmp;
}

__kernel void vectorTest2(const __global float *input, __global float *output,
                          uint width, uint height) {

  uint row = get_global_id(1);
  uint column = get_global_id(0);

  float tmp = 0.0f;

  for (uint index = 0; index < width; ++index)
    tmp += input[row + index];

  output[row * width + column] = tmp;
}
