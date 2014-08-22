__kernel void divRegion(__global float* input, __global float* output) {
  size_t globalId = get_global_id(0);
  float toStore = 0.f;

  if(input[globalId] > 0.5f)
    toStore = 10.f;
  else
    toStore = 2.f;

  output[globalId] = toStore;
}
