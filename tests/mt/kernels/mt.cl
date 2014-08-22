__kernel void mt(__global float *output, __global const float* input, int width, int height) {
  unsigned int row = get_global_id(1);
  unsigned int column = get_global_id(0);

  unsigned int indexIn  = row * width + column;
  unsigned int indexOut = column * height + row;
  output[indexOut] = input[indexIn]; 
}
