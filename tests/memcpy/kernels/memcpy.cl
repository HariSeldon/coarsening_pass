__kernel void rmrrmw(__global float* input, __global float* output, int size) {
  size_t globalId = get_global_id(0);

  for (unsigned int index = 0; index < size; ++index) {
    output[globalId * size + index] = input[globalId * size + index];
  }
}

__kernel void cmrcmw(__global float* input, __global float* output, int size) {
  size_t globalId = get_global_id(0);

  for (unsigned int index = 0; index < size; ++index) {
    output[index * size + globalId] = input[index * size + globalId];
  }
}
