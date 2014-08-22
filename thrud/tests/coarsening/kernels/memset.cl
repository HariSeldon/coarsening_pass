__kernel void memset1(__global float *output) {
  uint index = get_global_id(0);

  output[index] = 0.f;
}

__kernel void memset2(__global float *output, int width) {
  uint row = get_global_id(1);
  uint column = get_global_id(0);

  output[row * width + column] = 0.f;
}
