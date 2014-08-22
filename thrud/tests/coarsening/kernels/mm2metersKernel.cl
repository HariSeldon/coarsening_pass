__kernel void mm2metersKernel(__global float *depth, const uint depthSize,
                              const __global ushort *in, const uint inSize,
                              const int ratio) {
  uint x = get_global_id(0);
  uint y = get_global_id(1);

  depth[x + depthSize * y] =
      in[x * ratio + inSize * y * ratio] / 1000.0f;
}
