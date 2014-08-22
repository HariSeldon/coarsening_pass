__kernel void MatVecMulUncoalesced0(const __global float *M,
                                    const __global float *V, uint width,
                                    uint height, __global float *W) {
  uint y = get_global_id(0);
  float dotProduct = 0;
  for (int x = 0; x < width; ++x) {
    dotProduct += M[y * width + x] * V[x];
  }
  W[y] = dotProduct;
}

__kernel void MatVecMulUncoalesced1(const __global float *M,
                                    const __global float *V, uint width,
                                    uint height, __global float *W) {
  for (uint y = get_global_id(0); y < height; y += get_global_size(0)) {
    const __global float *row = M + y * width;
    float dotProduct = 0;
    for (uint x = 0; x < width; ++x)
      dotProduct += row[x] * V[x];
    W[y] = dotProduct;
  }
}

__kernel void MatVecMulCoalesced0(const __global float *M,
                                  const __global float *V, uint width,
                                  uint height, __global float *W,
                                  __local float *partialDotProduct) {
  for (uint y = get_group_id(0); y < height; y += get_num_groups(0)) {
    const __global float *row = M + y * width;
    float sum = 0;
    for (uint x = get_local_id(0); x < width; x += get_local_size(0))
      sum += row[x] * V[x];
    partialDotProduct[get_local_id(0)] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) == 0) {
      float dotProduct = 0;
      for (uint t = 0; t < get_local_size(0); ++t)
        dotProduct += partialDotProduct[t];
      W[y] = dotProduct;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}
