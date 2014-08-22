#ifndef KERNEL_H
#define KERNEL_H

#include <CL/cl.h>

class Buffer;
class Device;
class Program;

class Kernel {

// Constructors and Destructors.
//------------------------------------------------------------------------------
public:
  Kernel(const Program& program, const char* name);
  ~Kernel() throw();

// Public methods.
//------------------------------------------------------------------------------
public:
  cl_kernel getId() const;
  void setArgument(cl_uint index, size_t size, const void* pointer);
  void setArgument(cl_uint index, const Buffer& buffer);
  
  // OpenCL 1.2 only.
  //std::vector<size_t> getMaxGlobalWorkSize(const Device& device) const;
  size_t getMaxWorkGroupSize(const Device& device) const;
  unsigned long getLocalMemoryUsage(const Device& device) const;
  unsigned long getPrivateMemoryUsage(const Device& device) const;
  size_t getPreferredWorkGroupSizeMultiple(const Device& device) const;

// Private Fields.
//------------------------------------------------------------------------------
private:
  cl_kernel kernel;
};

// Traits.
//------------------------------------------------------------------------------
template <typename returnType> struct KernelInfoTraits {
  static returnType getKernelInfo(cl_kernel kernelId,
                                  const Device& device,
                                  cl_kernel_work_group_info kernelInfoName);
};

// OpenCL 1.2 only.
//template <> struct KernelInfoTraits<std::vector<size_t> > {
//  static std::vector<size_t> getKernelInfo(
//                             cl_kernel kernelId,
//                             const Device& device,
//                             cl_kernel_work_group_info kernelInfoName);
//};

#endif
