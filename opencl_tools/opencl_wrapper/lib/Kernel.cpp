#include "Kernel.h"

#include <CL/cl.h>

#include "Buffer.h"
#include "Device.h"
#include "Program.h"
#include "Utils.h"

//------------------------------------------------------------------------------
Kernel::Kernel(const Program& program, const char* name) {
  cl_int errorCode;
  kernel = clCreateKernel(program.getId(), name, &errorCode);  
  verifyOutputCode(errorCode, "Error creating the kernel");
}

//------------------------------------------------------------------------------
Kernel::~Kernel() throw() {
  clReleaseKernel(kernel);
}

//------------------------------------------------------------------------------
cl_kernel Kernel::getId() const {
  return kernel;
}

//------------------------------------------------------------------------------
void verifySetArgumentCode(int errorCode, unsigned int index) {
  if(isError(errorCode)) {
    std::string errorMessage = "Error setting kernel argument number " + 
                               index;
    throwException(errorCode, errorMessage.c_str());
  }
}

//------------------------------------------------------------------------------
void Kernel::setArgument(unsigned int index, size_t size, const void* pointer) {
  cl_int errorCode = clSetKernelArg(kernel, (cl_uint) index, size, pointer);
  verifySetArgumentCode(errorCode, index); 
}

//------------------------------------------------------------------------------
void Kernel::setArgument(unsigned int index, const Buffer& buffer) {
  cl_mem rawBuffer = buffer.getId();
  setArgument(index, sizeof(cl_mem), &rawBuffer);
}

//------------------------------------------------------------------------------
// OpenCL 1.2 only.
//std::vector<size_t> getMaxGlobalWorkSize(const Device& device) const {
//  return KernelInfoTraits<std::vector<size_t> >::getKernelInfo(
//         kernel, device, 
//         CL_KERNEL_GLOBAL_WORK_SIZE); 
//}

//------------------------------------------------------------------------------
size_t Kernel::getMaxWorkGroupSize(const Device& device) const {
  return KernelInfoTraits<size_t>::getKernelInfo(kernel, device,
                                                 CL_KERNEL_WORK_GROUP_SIZE);
}

//------------------------------------------------------------------------------
unsigned long Kernel::getLocalMemoryUsage(const Device& device) const {
  return KernelInfoTraits<cl_ulong>::getKernelInfo(kernel, device,
                                                   CL_KERNEL_LOCAL_MEM_SIZE);
}

//------------------------------------------------------------------------------
unsigned long Kernel::getPrivateMemoryUsage(const Device& device) const {
  return KernelInfoTraits<cl_ulong>::getKernelInfo(kernel, device,
                                                   CL_KERNEL_PRIVATE_MEM_SIZE);
}

//------------------------------------------------------------------------------
size_t Kernel::getPreferredWorkGroupSizeMultiple(const Device& device) const {
  return KernelInfoTraits<size_t>::getKernelInfo(
         kernel, device,
         CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE);
}

//------------------------------------------------------------------------------
template <typename returnType>
returnType KernelInfoTraits<returnType>::getKernelInfo(
           cl_kernel kernelId,
           const Device& device,
           cl_kernel_work_group_info kernelInfoName) {
  returnType result;
  cl_int errorCode = clGetKernelWorkGroupInfo(kernelId, device.getId(),
                                              kernelInfoName,
                                              sizeof(returnType), 
                                              &result, NULL);
  verifyOutputCode(errorCode, "Error querying kernel info: ");
  return result;
}

//------------------------------------------------------------------------------
size_t getKernelInfoSize(
       cl_kernel kernelId,
       cl_device_id deviceId,
       cl_kernel_work_group_info kernelInfoName) {
  size_t result;
  cl_int errorCode = clGetKernelWorkGroupInfo(kernelId, deviceId,
                                              kernelInfoName,
                                              sizeof(result), 
                                              NULL, &result);
  verifyOutputCode(errorCode, "Error querying device info size: ");
  return result;
}

//------------------------------------------------------------------------------
// OpenCL 1.2 only.
//std::vector<size_t> KernelInfoTraits<std::vector<size_t> >::getKernelInfo(
//                    cl_kernel kernelId,
//                    const Device& device,
//                    cl_kernel_work_group_info kernelInfoName) {
//  size_t resultSize = getKernelInfoSize(deviceId, kernelInfoName);
//  size_t* rawResult = new size_t[resultSize];
//  cl_int errorCode = clGetKernelWorkGroupInfo(kernelId, device.getId(),
//                                              resultSize, rawResult, NULL);
//  std::vector<size_t> result(rawResult, rawResult + resultSize); 
//  delete [] rawResult;
//  return result; 
//}
