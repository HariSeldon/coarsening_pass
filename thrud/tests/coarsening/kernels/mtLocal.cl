// Don't touch this version.
//__kernel void mtLocal(__global float *odata, __global float *idata, int width, int height, __local float* block)
//{
//  unsigned int localHeight = get_local_size(1);
//  unsigned int localWidth = get_local_size(0);
//
//  unsigned int groupRow = get_group_id(1);
//  unsigned int groupColumn = get_group_id(0);
//
//  unsigned int row = get_global_id(1);
//  unsigned int column = get_global_id(0);
//  
//  unsigned int localRow = get_local_id(1);
//  unsigned int localColumn = get_local_id(0);
//
//  unsigned int index_in = row * width + column;
//  block[localRow * (localWidth) + localColumn] = idata[index_in];
//
//  barrier(CLK_LOCAL_MEM_FENCE);
//
//  unsigned int Z = localWidth * groupColumn * height + localHeight * groupRow;
//  unsigned int index_out = Z + height * localColumn + localRow;
//  odata[index_out] = block[localRow * localWidth + localColumn];
//}

__kernel void mtLocal(__global float *odata, __global float *idata, int width, int height, __local float* block)
{
  unsigned int localHeight = get_local_size(1);
  unsigned int localWidth = get_local_size(0);

  unsigned int groupRow = get_group_id(1);
  unsigned int groupColumn = get_group_id(0);

  unsigned int row = get_global_id(1);
  unsigned int column = get_global_id(0);

  unsigned int localRow = get_local_id(1);
  unsigned int localColumn = get_local_id(0);

  unsigned int index_in = row * width + column;
  block[localRow * (localWidth) + localColumn] = idata[index_in];

  barrier(CLK_LOCAL_MEM_FENCE);

  unsigned int Z = localWidth * groupColumn * height + localHeight * groupRow;

  unsigned int localIndex = localRow * localWidth + localColumn;
  unsigned int tRow = localIndex / localHeight;
  unsigned int tColumn = localIndex % localHeight;

  unsigned int index_out = Z + height * tRow + tColumn;
  odata[index_out] = block[tColumn * localWidth + tRow];
}

__kernel void mt(__global float *odata, __global const float* idata, int width, int height) {
  unsigned int row = get_global_id(1);
  unsigned int column = get_global_id(0);

  unsigned int indexIn  = row * width + column;
  unsigned int indexOut = column * height + row;
  odata[indexOut] = idata[indexIn]; 
}

//__kernel void mtLocal(__global float *odata, __global float *idata, int width, int height, __local float* block)
//{
//  unsigned int localHeight = get_local_size(0);
//  unsigned int localWidth = get_local_size(1);
//  // read the matrix tile into shared memory
//  unsigned int xIndex = get_global_id(0);
//  unsigned int yIndex = get_global_id(1);
//
//  if((xIndex < width) && (yIndex < height))
//  {
//    unsigned int index_in = yIndex * width + xIndex;
//    block[get_local_id(1)*(localWidth+1)+get_local_id(0)] = idata[index_in];
//  }
//
//  barrier(CLK_LOCAL_MEM_FENCE);
//
//  // write the transposed matrix tile to global memory
//  xIndex = get_group_id(1) * localWidth + get_local_id(0);
//  yIndex = get_group_id(0) * localHeight + get_local_id(1);
//  if((xIndex < height) && (yIndex < width))
//    {
//    unsigned int index_out = yIndex * height + xIndex;
//    odata[index_out] = block[get_local_id(0)*(localWidth+1)+get_local_id(1)];
//  }
//}

//__kernel void mtLocal(__global float *odata, __global float *idata, int width, int height, __local float* block) {
//  unsigned int row = get_global_id(1);
//  unsigned int column = get_global_id(0);
//
//  unsigned int localRow = get_local_id(1);
//  unsigned int localColumn = get_local_id(0);
//
//  unsigned int localWidth = get_local_size(1);
//  unsigned int localHeight = get_local_size(0);
//
//  unsigned int groupRow = get_group_id(1);
//  unsigned int groupColumn = get_group_id(0);
//
//  if((column < width) && (row < height)) {
//    unsigned int element = row * width + column;
//    block[localRow * (localWidth + 1) + localColumn] = idata[element];
//  }
//
//  barrier(CLK_LOCAL_MEM_FENCE);
//
//  if((row < width) && (column < height)) {
////    unsigned int indexOut = column * height + row;
////    unsigned int localIndex = localRow * (localWidth + 1) + localColumn;
//    unsigned int indexOut = row * height + column;
//    unsigned int localIndex = localColumn * (localWidth + 1) + localRow;
//    odata[indexOut] = block[localIndex];
//  }
//}

//#define BLOCK_DIM 16

// This kernel is optimized to ensure all global reads and writes are coalesced,
// and to avoid bank conflicts in shared memory.  This kernel is up to 11x faster
// than the naive kernel below.  Note that the shared memory array is sized to 
// (BLOCK_DIM+1)*BLOCK_DIM.  This pads each row of the 2D block in shared memory 
// so that bank conflicts do not occur when threads address the array column-wise.
