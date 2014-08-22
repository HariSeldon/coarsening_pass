#ifndef BUFFER_H
#define BUFFER_H

#include <CL/cl.h>

class Context;

class Buffer {

public:
  enum MemoryFlags {
    ReadWrite = CL_MEM_READ_WRITE,
    WriteOnly = CL_MEM_WRITE_ONLY,
    ReadOnly = CL_MEM_READ_ONLY,
    UseHostPtr = CL_MEM_USE_HOST_PTR,
    AllocHostPtr = CL_MEM_ALLOC_HOST_PTR,
    CopyHostPtr = CL_MEM_COPY_HOST_PTR
  };

// Constructors and Destructors.
//------------------------------------------------------------------------------
public:
  Buffer(const Context& context, MemoryFlags flags, size_t size, 
         void *host_ptr);
  ~Buffer() throw();

public:
  cl_mem getId() const;

private:
  cl_mem buffer;
  size_t size;
};

#endif
