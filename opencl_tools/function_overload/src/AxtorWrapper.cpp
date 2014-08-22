#include <CL/cl.h>

#include "Utils.h"

#include <stdlib.h>
#include <sys/types.h>
#include <dlfcn.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <stdexcept>
#include <vector>

std::string compile(std::string &inputFile, const char *options,
                    std::string &outputFile, int seed);

//------------------------------------------------------------------------------
// OpenCL Runtime state data structures.
struct ProgramDesc {
  cl_context context;
  std::string sourceStr;
  cl_program handle;

  ProgramDesc(cl_context context, std::string sourceStr)
      : context(context), sourceStr(sourceStr), handle(0) {}

  ProgramDesc(cl_context context, cl_program handle)
      : context(context), sourceStr(""), handle(handle) {}

  bool isValid() const { return handle != 0; }
  bool isFromBinary() const { return sourceStr.empty(); }
};

typedef std::vector<ProgramDesc *> ProgramDescVec;

// Runtime state state
static ProgramDescVec programs;

// OpenCL functions.
//------------------------------------------------------------------------------

extern "C" cl_kernel clCreateKernel(cl_program program, const char *kernel_name,
                                    cl_int *errcode_ret) {
  ProgramDesc *desc = reinterpret_cast<ProgramDesc *>(program);
  clCreateKernelFunction originalCreateKernel;
  *(void **)(&originalCreateKernel) = dlsym(RTLD_NEXT, CL_CREATE_KERNEL_NAME);

  cl_int errorCode;
  cl_kernel kernel =
      originalCreateKernel(desc->handle, kernel_name, &errorCode);
  dumpError(errorCode);
  if (errcode_ret)
    *errcode_ret = errorCode;

  return kernel;
}

//------------------------------------------------------------------------------
extern "C" cl_int clReleaseProgram(cl_program program) {
  ProgramDesc *desc = reinterpret_cast<ProgramDesc *>(program);

  clReleaseProgramFunction originalReleaseProgram;
  *(void **)(&originalReleaseProgram) =
      dlsym(RTLD_NEXT, CL_RELEASE_PROGRAM_NAME);

  return dumpError(originalReleaseProgram(desc->handle));
}

//------------------------------------------------------------------------------
extern "C" cl_int clRetainProgram(cl_program program) {
  ProgramDesc *desc = reinterpret_cast<ProgramDesc *>(program);

  clRetainProgramFunction originalRetainProgram;
  *(void **)(&originalRetainProgram) = dlsym(RTLD_NEXT, CL_RETAIN_PROGRAM_NAME);

  return dumpError(originalRetainProgram(desc->handle));
}

//------------------------------------------------------------------------------
extern "C" cl_int clGetProgramInfo(cl_program program,
                                   cl_program_info param_name,
                                   size_t param_value_size, void *param_value,
                                   size_t *param_value_size_ret) {
  ProgramDesc *desc = reinterpret_cast<ProgramDesc *>(program);

  clGetProgramInfoFunction originalGetProgramInfo;
  *(void **)(&originalGetProgramInfo) =
      dlsym(RTLD_NEXT, CL_GET_PROGRAM_INFO_NAME);
  return originalGetProgramInfo(desc->handle, param_name, param_value_size,
                                param_value, param_value_size_ret);
}

//------------------------------------------------------------------------------
extern "C" cl_int clGetProgramBuildInfo(cl_program program, cl_device_id device,
                                        cl_program_build_info param_name,
                                        size_t param_value_size,
                                        void *param_value,
                                        size_t *param_value_size_ret) {
  ProgramDesc *desc = reinterpret_cast<ProgramDesc *>(program);

  clGetProgramBuildInfoFunction originalGetProgramBuildInfo;
  *(void **)(&originalGetProgramBuildInfo) =
      dlsym(RTLD_NEXT, CL_GET_PROGRAM_BUILD_INFO_NAME);
  return dumpError(originalGetProgramBuildInfo(desc->handle, device, param_name,
                                               param_value_size, param_value,
                                               param_value_size_ret));
}

//------------------------------------------------------------------------------
extern "C" cl_program
clCreateProgramWithBinary(cl_context context, cl_uint num_devices,
                          const cl_device_id *device_list,
                          const size_t *lengths, const unsigned char **binaries,
                          cl_int *binary_status, cl_int *errcode_ret) {
  clCreateProgramWithBinaryFunction originalCreateProgramWithBinary;
  *(void **)(&originalCreateProgramWithBinary) =
      dlsym(RTLD_NEXT, CL_CREATE_PROGRAM_WITH_BINARY_NAME);
  cl_program realHandle = originalCreateProgramWithBinary(
      context, num_devices, device_list, lengths, binaries, binary_status,
      errcode_ret);

  ProgramDesc *desc = new ProgramDesc(context, realHandle);
  programs.push_back(desc);
  return reinterpret_cast<cl_program>(desc);
}

//------------------------------------------------------------------------------
extern "C" cl_program clCreateProgramWithSource(cl_context context,
                                                cl_uint count,
                                                const char **strings,
                                                const size_t *,
                                                cl_int *errcode_ret) {
  std::cout << "HIJACKED clCreateProgramWithSource HIJACKED\n";
  std::stringstream buffer;
  for (uint i = 0; i < count; ++i) {
    buffer << strings[i] << "\n";
  }

  ProgramDesc *desc = new ProgramDesc(context, buffer.str());
  programs.push_back(desc);
  cl_program fakeHandle = reinterpret_cast<cl_program>(desc);

  if (errcode_ret)
    *errcode_ret = CL_SUCCESS;

  return fakeHandle;
}

//------------------------------------------------------------------------------
extern "C" cl_int clBuildProgram(cl_program program, cl_uint num_devices,
                                 const cl_device_id *device_list,
                                 const char *options,
                                 void (*pfn_notify)(cl_program, void *),
                                 void *user_data) {
  std::cout << "HIJACKED clBuildProgram HIJACKED\n";

  srand((unsigned)time(0));
  int seed = rand() % 100000;

  // Get pointer to original function call.
  clBuildProgramFunction originalBuildProgram;
  *(void **)(&originalBuildProgram) = dlsym(RTLD_NEXT, CL_BUILD_PROGRAM_NAME);

  clCreateProgramWithSourceFunction originalCreateProgramWithSource;
  *(void **)(&originalCreateProgramWithSource) =
      dlsym(RTLD_NEXT, CL_CREATE_PROGRAM_WITH_SOURCE_NAME);

  // Get the source file name.
  std::string inputFile = getMangledFileName(OCL_INPUT_FILE, seed);
  std::string outputFile = getMangledFileName(OCL_OUTPUT_FILE, seed);

  // Get the program handle.
  ProgramDesc *desc = reinterpret_cast<ProgramDesc *>(program);
  if (desc->isFromBinary()) {
    std::cout << "OpenCL function overloading does not work when creating "
                 "the program with binary\n";
    exit(1);
  }

  // Dump the program.
  writeFile(inputFile, desc->sourceStr);

  // Compile the program.
  std::string oclOptions = compile(inputFile, options, outputFile, seed);

  // Create the new program.
  size_t outputSize;
  const char *outputProgram = readFile(outputFile.c_str(), &outputSize);
  cl_int errorCode;

  desc->handle = originalCreateProgramWithSource(
      desc->context, 1, (const char **)&outputProgram, &outputSize, &errorCode);
  verifyOutputCode(errorCode, "Error creating the new program");

  // Build the new program.
  errorCode = originalBuildProgram(desc->handle, num_devices, device_list,
                                   oclOptions.c_str(), pfn_notify, user_data);
  verifyOutputCode(errorCode, "Error building the new program");

  std::string removeString = "rm " + inputFile + " && rm " + outputFile;
  system(removeString.c_str());

  delete[] outputProgram;

  return CL_SUCCESS;
}

//------------------------------------------------------------------------------
std::string compile(std::string &inputFile, const char *options,
                    std::string &outputFile, int seed) {
  // Compile the program.
  std::string optOptions = getEnvString(OCL_COMPILER_OPTIONS);

  if (options == NULL)
    options = "";

  std::string clangOptions(options);
  std::string oclOptions;

  splitCompilerOptions(clangOptions, oclOptions);

  if (compileWithAxtor(inputFile, clangOptions, optOptions, outputFile, seed)) {
    std::cout << "Error compiling with axtor\n";
    exit(1);
  }

  return oclOptions;
}

//------------------------------------------------------------------------------
cl_int clEnqueueNDRangeKernel(
    cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim,
    const size_t *global_work_offset, const size_t *global_work_size,
    const size_t *local_work_size, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event) {
  std::cerr << "HIJACKED clEnqueueNDRangeKernel HIJACKED\n";

  // Setup the event to measure the kernel execution time.
  bool isEventNull = false;
  if (event == NULL) {
    isEventNull = true;
    event = new cl_event();
  }

  std::string kernelName = getKernelName(kernel);
  std::string envKernelName = getEnvString("TC_KERNEL_NAME");

  size_t *newGlobalSize = new size_t[work_dim];
  size_t *newLocalSize = new size_t[work_dim];

  if (kernelName != envKernelName) {
    std::cout << "No coarsening for: " << kernelName << "\n";
    std::cout << "gws " << work_dim << " " << global_work_size[0] << "\n";
    memcpy(newGlobalSize, global_work_size, work_dim * sizeof(size_t));

    if (local_work_size != NULL)
      memcpy(newLocalSize, local_work_size, work_dim * sizeof(size_t));
    else
      newLocalSize = NULL;
  } else {
    bool NDRangeResult =
        computeNDRangeDim(work_dim, global_work_size, local_work_size,
                          newGlobalSize, newLocalSize);

    if (NDRangeResult == false) {
      if (memcmp(global_work_size, newGlobalSize, work_dim * sizeof(size_t)) ==
          0) {
        newLocalSize = NULL;
      } else {
        std::cout << "Cannot apply coarsening when local work size is null\n";
        return 1;
      }
    }
  }

  std::string repetitionsString = getEnvString(OCL_REPETITIONS);
  unsigned int repetitions;
  if (repetitionsString != "")
    std::istringstream(repetitionsString) >> repetitions;
  else
    repetitions = 1;

  enqueueKernel(command_queue, kernel, work_dim, global_work_offset,
                newGlobalSize, newLocalSize, num_events_in_wait_list,
                event_wait_list, event, repetitions, kernelName);

  if (isEventNull) {
    clReleaseEvent(*event);
    delete event;
    event = NULL;
  }

  delete[] newGlobalSize;
  delete[] newLocalSize;

  return CL_SUCCESS;
}

//------------------------------------------------------------------------------
cl_command_queue clCreateCommandQueue(cl_context context, cl_device_id device,
                                      cl_command_queue_properties properties,
                                      cl_int *errcode_ret) {
  std::cout << "HIJACKED clCreateCommandQueue HIJACKED\n";

  // Get pointer to original function calls.
  clCreateCommandQueueFunction originalclCreateCommandQueue;
  *(void **)(&originalclCreateCommandQueue) =
      dlsym(RTLD_NEXT, CL_CREATE_COMMAND_QUEUE_NAME);

  properties = properties | CL_QUEUE_PROFILING_ENABLE;

  return originalclCreateCommandQueue(context, device, properties, errcode_ret);
}
