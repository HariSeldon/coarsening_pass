#include "Buffer.h"

#include "Context.h"
#include "Utils.h"

Buffer::Buffer(const Context& context, MemoryFlags flags, size_t size, 
               void* hostPtr) : size(size) {
  cl_int errorCode;
  buffer = clCreateBuffer(context.getId(), flags, size, hostPtr, 
                          &errorCode);
  verifyOutputCode(errorCode, "Error creating the buffer");
}

Buffer::~Buffer() throw() { 
  clReleaseMemObject(buffer);
}

cl_mem Buffer::getId() const {
  return buffer;
}
