__kernel void memset2D(__global float* input, int width) {
  size_t column = get_global_id(0);
  size_t row = get_global_id(1);

  input[row * width + column] = CONSTANT;
}
