__kernel void mt(__global float *odata, __global const float* idata, int width, int height) {
  unsigned int row = get_global_id(1);
  unsigned int column = get_global_id(0);

  unsigned int indexIn  = row * width + column;
  unsigned int indexOut = column * height + row;
  odata[indexOut] = idata[indexIn];
}
