#include <Utils.h>

#include <CL/cl.h>

#include <stdlib.h>
#include <sys/types.h>
#include <dlfcn.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

// Enqueue overloading.
//------------------------------------------------------------------------------
cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue,
                              cl_kernel kernel,
                              cl_uint work_dim,
                              const size_t* global_work_offset,
                              const size_t* global_work_size,
                              const size_t* local_work_size,
                              cl_uint num_events_in_wait_list,
                              const cl_event* event_wait_list,
                              cl_event* event) {
  std::cerr << "HIJACKED clEnqueueNDRangeKernel HIJACKED\n";

  // Setup the event to measure the kernel execution time.
  bool isEventNull = (event == NULL);
  if(isEventNull) {
    event = new cl_event();
    clRetainEvent(*event);
  }

  std::string kernelName = getKernelName(kernel);

  size_t* newGlobalSize = new size_t [work_dim];
  size_t* newLocalSize = new size_t [work_dim];

  std::string repetitionsString = getEnvString(OCL_REPETITIONS);
  unsigned int repetitions;
  if(repetitionsString != "")
    std::istringstream(repetitionsString) >> repetitions;
  else
    repetitions = 1;

  enqueueKernel(command_queue, kernel, work_dim, global_work_offset,
                global_work_size, local_work_size, num_events_in_wait_list,
                event_wait_list, event, repetitions, kernelName);

  if(isEventNull) {
    clReleaseEvent(*event);
    delete event;
  }

  delete [] newGlobalSize;
  delete [] newLocalSize;

  return CL_SUCCESS;
}

//------------------------------------------------------------------------------
cl_command_queue clCreateCommandQueue(cl_context context,
                                      cl_device_id device,
                                      cl_command_queue_properties properties,
                                      cl_int* errcode_ret) {
  std::cout << "HIJACKED clCreateCommandQueue HIJACKED\n";

  // Get pointer to original function calls.
  clCreateCommandQueueFunction originalclCreateCommandQueue;
    *(void **)(&originalclCreateCommandQueue) =
    dlsym(RTLD_NEXT, CL_CREATE_COMMAND_QUEUE_NAME);

  properties = properties | CL_QUEUE_PROFILING_ENABLE;

  return originalclCreateCommandQueue(context, device, properties, errcode_ret);
}
