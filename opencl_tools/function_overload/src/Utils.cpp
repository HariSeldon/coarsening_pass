#include "Utils.h"

#include <CL/cl.h>
#include <algorithm>
#include <stdlib.h>
#include <sys/types.h>
#include <dlfcn.h>
#include <iostream>
#include <iterator>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string.h>
#include <vector>

#define RELAXED_MATH "-cl-fast-relaxed-math"
#define OCL_OPTIONS_NUMBER 1

const char *oclOptionsList[OCL_OPTIONS_NUMBER] = {RELAXED_MATH};

// System Functions.
//------------------------------------------------------------------------------
char *readFile(const char *filePath, size_t *size) {
  std::ifstream fileStream(filePath);
  if (fileStream.is_open()) {
    fileStream.seekg(0, std::ios::end);
    int fileSize = fileStream.tellg();
    char *source = new char[fileSize + 1];
    fileStream.seekg(0, std::ios::beg);
    fileStream.read(source, fileSize);
    fileStream.close();
    source[fileSize] = '\0';
    *size = fileSize + 1;
    return source;
  } else {
    std::stringstream ss;
    ss << "Cannot open: " << filePath;
    throw std::runtime_error(ss.str());
  }
}

//------------------------------------------------------------------------------
void writeFile(std::string filePath, const std::string &data) {
  std::ofstream out(filePath.c_str());
  out << data;
  out.flush();
  out.close();
}

//------------------------------------------------------------------------------
cl_int dumpError(cl_int errorCode) {
  if (errorCode == CL_SUCCESS)
    return errorCode;

  std::string name;
  switch (errorCode) {
#define NAMED_ERROR(CODE)                                                      \
  case CODE:                                                                   \
    name = #CODE;                                                              \
    break;
    NAMED_ERROR(CL_INVALID_CONTEXT)
    NAMED_ERROR(CL_INVALID_COMMAND_QUEUE)
    NAMED_ERROR(CL_BUILD_ERROR)
    NAMED_ERROR(CL_BUILD_PROGRAM_FAILURE)
    NAMED_ERROR(CL_INVALID_ARG_INDEX)
    NAMED_ERROR(CL_INVALID_BINARY)
    NAMED_ERROR(CL_INVALID_BUILD_OPTIONS)
    NAMED_ERROR(CL_INVALID_PROGRAM)
    NAMED_ERROR(CL_INVALID_PROGRAM_EXECUTABLE)
    NAMED_ERROR(CL_INVALID_KERNEL)
    NAMED_ERROR(CL_INVALID_KERNEL_DEFINITION)
    NAMED_ERROR(CL_INVALID_KERNEL_NAME)
#undef NAMED_ERROR
  default:
    name = "<unnamed>";
    break;
  };

  std::cout << "Failed with " << name << "!!!\n";
  return errorCode;
}

//------------------------------------------------------------------------------
std::string getEnvString(const char *name, const char *defValue) {
  char *val = getenv(name);
  return std::string(val ? val : defValue);
}

// OpenCL functions.
//------------------------------------------------------------------------------
cl_device_id getDeviceFromContext(cl_context context,
                                  unsigned int deviceNumber) {
  int errorCode;
  cl_device_id *devices;
  size_t devicesNumber;
  errorCode = clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(size_t),
                               &devicesNumber, NULL);
  verifyOutputCode(errorCode, "Error getting device number from context");

  devices = new cl_device_id[devicesNumber];
  errorCode =
      clGetContextInfo(context, CL_CONTEXT_DEVICES,
                       sizeof(cl_device_id) * devicesNumber, devices, NULL);
  verifyOutputCode(errorCode, "Error getting devices from context");
  return devices[deviceNumber];
}

//------------------------------------------------------------------------------
char *getProgramSourceCode(cl_program program, size_t *codeSize) {
  size_t tmpSize;

  cl_int errorCode =
      clGetProgramInfo(program, CL_PROGRAM_SOURCE, 0, NULL, &tmpSize);
  verifyOutputCode(errorCode, "Error querying the source code size");

  char *sourceCode = new char[tmpSize];
  errorCode =
      clGetProgramInfo(program, CL_PROGRAM_SOURCE, tmpSize, sourceCode, NULL);
  verifyOutputCode(errorCode, "Error querying the source code");
  *codeSize = tmpSize;
  return sourceCode;
}

//------------------------------------------------------------------------------
cl_device_id *getProgramDevices(cl_program program,
                                unsigned int *devicesNumber) {
  cl_device_id *devicesData;
  // Query the devices number.
  cl_int errorCode = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES,
                                      sizeof(cl_uint), devicesNumber, NULL);
  verifyOutputCode(errorCode, "Error getting number of devices associated "
                              "with the program");
  // Query the devices.
  devicesData = new cl_device_id[*devicesNumber];
  errorCode = clGetProgramInfo(program, CL_PROGRAM_DEVICES,
                               sizeof(cl_device_id) * (*devicesNumber),
                               devicesData, NULL);
  verifyOutputCode(errorCode, "Error getting devices associated with the "
                              "program");
  return devicesData;
}

//------------------------------------------------------------------------------
cl_context getProgramContext(cl_program program) {
  cl_context result;
  cl_int errorCode = clGetProgramInfo(
      program, CL_PROGRAM_CONTEXT, sizeof(cl_context), (void *)&result, NULL);
  verifyOutputCode(errorCode, "Error getting program context");
  return result;
}

//------------------------------------------------------------------------------
cl_context getKernelContext(cl_kernel kernel) {
  cl_context result;
  cl_int errorCode = clGetKernelInfo(kernel, CL_KERNEL_CONTEXT,
                                     sizeof(cl_context), (void *)&result, NULL);
  verifyOutputCode(errorCode, "Error getting kernel context");
  return result;
}

//------------------------------------------------------------------------------
unsigned long int computeEventDuration(cl_event *event) {
  if (event == NULL)
    throw std::runtime_error("Error computing event duration. \
                              Event is not initialized");
  cl_int errorCode;
  cl_ulong start, end;
  errorCode = clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_START,
                                      sizeof(cl_ulong), &start, NULL);
  verifyOutputCode(errorCode, "Error querying the event start time");
  errorCode = clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_END,
                                      sizeof(cl_ulong), &end, NULL);
  verifyOutputCode(errorCode, "Error querying the event end time");
  return static_cast<unsigned long>(end - start);
}

// Error checking functions.
//------------------------------------------------------------------------------
inline bool isError(cl_int valueToCheck) { return valueToCheck != CL_SUCCESS; }

//------------------------------------------------------------------------------
void verifyOutputCode(cl_int valueToCheck, const char *errorMessage) {
  if (isError(valueToCheck)) {
    std::cout << errorMessage << " " << valueToCheck << "\n";
    exit(valueToCheck);
  }
}

//------------------------------------------------------------------------------
std::string getMangledFileName(const char *fileName, int seed) {
  std::stringstream stream;
  stream << fileName << seed;
  return stream.str();
}

//------------------------------------------------------------------------------
std::string getKernelName(cl_kernel kernel) {
  size_t nameSize;
  cl_int errorCode =
      clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 0, NULL, &nameSize);
  verifyOutputCode(errorCode, "Error querying the kernel name size");
  char *name = new char[nameSize];

  errorCode =
      clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, nameSize, name, NULL);
  verifyOutputCode(errorCode, "Error querying the kernel name");

  std::string nameString(name);
  delete[] name;

  return nameString;
}

//------------------------------------------------------------------------------
int compileWithAxtor(std::string &inputFile, std::string &clangOptions,
                     std::string &optOptions, std::string &outputFile,
                     int seed) {
  std::string bitcodeFile = getMangledFileName(BC_FILE, seed);

  // Inline commands.
  std::string sedCmdOne = "sed \'s/__inline/inline/\' -i " + inputFile;
  std::string sedCmdTwo = "sed \'s/inline/static inline/\' -i " + inputFile;

  std::string oclHeader = getEnvString("OCL_HEADER");

  // WARNING: be sure to redirect the stderr only to /dev/null.
  // Clang command.
  std::string clangCmd = "LD_PRELOAD=\"\" clang -x cl -target spir -include " +
                         oclHeader + " -O0 " + clangOptions + " " + inputFile +
                         " -S -emit-llvm -fno-builtin -o " + bitcodeFile +
                         " 2> /dev/null";

  // Opt command.
  std::string optCmd = "LD_PRELOAD=\"\" opt " + optOptions + " " + bitcodeFile +
                       " -o " + bitcodeFile + " 2> /dev/null";

  // Axtor command.
  std::string axtorCmd = "LD_PRELOAD=\"\" axtor " + bitcodeFile + " -m OCL " +
                         "-o " + outputFile + " 2> /dev/null";

  std::cout << "CLANG:\n" << clangCmd << "\nOPT:\n" << optCmd << "\nAXTOR:\n"
            << axtorCmd << "\n";

  // Inline.
  system(sedCmdOne.c_str());
  system(sedCmdTwo.c_str());

  // Clang.
  if (system(clangCmd.c_str())) {
    std::cout << "&&&&& FRONTEND_FAILURE!";
    return 1;
  }

  // Opt.
  if (system(optCmd.c_str())) {
    std::cout << "&&&&& OPT_FAILURE!";
    return 2;
  }

  // Axtor.
  if (system(axtorCmd.c_str())) {
    std::cout << "&&&&& AXTOR_FAILURE!";
    return 3;
  }

  // size_t bb;
  // char *tmp;
  // tmp = readFile(outputFile.c_str(), &bb);
  // std::cout << tmp << "\n";

  // Remove the bitcode file.
  std::string remove = "rm " + bitcodeFile;
  system(remove.c_str());

  return 0;
}

//------------------------------------------------------------------------------
bool computeNDRangeDim(unsigned int dimensions, const size_t *globalSize,
                       const size_t *localSize, size_t *newGlobalSize,
                       size_t *newLocalSize) {
  if (dimensions >= 4) {
    std::cout << "4 or more dimensions are not supported by the wrapper.\n";
    exit(1);
  }

  std::string compilerOptions = getEnvString(OCL_COMPILER_OPTIONS);
  unsigned int CF = 0;
  unsigned int CD = 0;
  std::pair<unsigned int, unsigned int> cp =
      getCoarseningOptions(compilerOptions);

  CF = cp.first;
  CD = cp.second;

  if (CF == 0 && CD == 0) {
    cp = getVectorizationOptions(compilerOptions);
    CF = cp.first;
    CD = cp.second;
  }

  if (CF == 0 && CD == 0) {
    CF = 1;
  }

  if (localSize == NULL && CF != 1) {
    std::cout << "Cannot apply coarsening when localSize is NULL.\n";
    return false;
  }

  if (localSize == NULL && CF == 1) {
    memcpy(newGlobalSize, globalSize, dimensions * sizeof(size_t));
    return false;
  }

  memcpy(newGlobalSize, globalSize, dimensions * sizeof(size_t));
  memcpy(newLocalSize, localSize, dimensions * sizeof(size_t));

  // If the CD is higher than what it should be than do nothing.
  if (CF != 1) {
    if (CD >= dimensions) {
      std::cout << "Error specifying a coarsening direction higher "
                   "than the number of dimensions.\n";
      newGlobalSize[CD] = globalSize[CD];
      newLocalSize[CD] = localSize[CD];
    }
    newGlobalSize[CD] = globalSize[CD] / CF;
    newLocalSize[CD] = localSize[CD] / CF;

    if (newGlobalSize[CD] == 0 || newLocalSize[CD] == 0) {
      newGlobalSize[CD] = globalSize[CD];
      newLocalSize[CD] = localSize[CD];
    }
  }

  return true;
}

//------------------------------------------------------------------------------
void enqueueKernel(cl_command_queue command_queue, cl_kernel kernel,
                   cl_uint work_dim, const size_t *global_work_offset,
                   const size_t *global_work_size,
                   const size_t *local_work_size,
                   cl_uint num_events_in_wait_list,
                   const cl_event *event_wait_list, cl_event *event,
                   unsigned int repetitions, const std::string &kernelName) {

  // Get pointer to original function calls.
  clEnqueueNDRangeKernelFunction originalclEnqueueKernel;
  *(void **)(&originalclEnqueueKernel) =
      dlsym(RTLD_NEXT, CL_ENQUEUE_NDRANGE_KERNEL_NAME);

  cl_int errorCode = 0;

  for (unsigned int index = 0; index < work_dim; ++index) {
    std::cout << "gs[" << index << "] = " << global_work_size[index] << "\n";
    if (local_work_size != NULL)
      std::cout << "ls[" << index << "] = " << local_work_size[index] << "\n";
    else
      std::cout << "ls = NULL\n";
  }

  for (unsigned int index = 0; index < repetitions; ++index) {
    errorCode = originalclEnqueueKernel(
        command_queue, kernel, work_dim, global_work_offset, global_work_size,
        local_work_size, num_events_in_wait_list, event_wait_list, event);
    verifyOutputCode(errorCode, "Error enqueuing the original kernel");
    clFinish(command_queue);
    cl_int eventStatus = clWaitForEvents(1, event);
    if (eventStatus == -5)
      std::cerr << kernelName + " 0\n";
    else
      std::cerr << kernelName << " " << computeEventDuration(event) << "\n";
    verifyOutputCode(errorCode, "Error releasing the event");
  }
}

//------------------------------------------------------------------------------
void splitCompilerOptions(std::string &clangOptions, std::string &oclOptions) {
  for (unsigned int index = 0; index < OCL_OPTIONS_NUMBER; ++index) {
    const char *oclOption = oclOptionsList[index];
    size_t startingPosition = clangOptions.find(oclOption);

    if (startingPosition != std::string::npos) {
      unsigned int length = strlen(oclOption);
      // Add the option to oclOptions.
      oclOptions += oclOption + std::string(" ");
      // Remove the option from clangOptions.
      clangOptions.erase(startingPosition, length);
    }
  }
}

//------------------------------------------------------------------------------
std::pair<unsigned int, unsigned int>
getTransformationOptions(const std::string &compilerOptions,
                         const std::string &factor,
                         const std::string &direction) {
  std::istringstream iss(compilerOptions);
  std::vector<std::string> tokens;
  std::copy(std::istream_iterator<std::string>(iss),
            std::istream_iterator<std::string>(),
            std::back_inserter<std::vector<std::string>>(tokens));

  std::vector<std::string>::const_iterator cfI =
      std::find(tokens.begin(), tokens.end(), factor);
  std::vector<std::string>::const_iterator cdI =
      std::find(tokens.begin(), tokens.end(), direction);
  if (cfI == tokens.end() && cdI == tokens.end())
    return std::make_pair(0, 0);

  if ((cfI != tokens.end() && cdI == tokens.end()) ||
      (cfI == tokens.end() && cdI != tokens.end())) {
    std::cout << "Error specifying the coarsening options.";
    exit(1);
  }

  unsigned int CD, CF = 0;

  cfI++;
  cdI++;
  std::stringstream(*cfI) >> CF;
  std::stringstream(*cdI) >> CD;

  return std::make_pair(CF, CD);
}

//------------------------------------------------------------------------------
// Returns cf/cd.
std::pair<unsigned int, unsigned int>
getCoarseningOptions(const std::string &compilerOptions) {
  return getTransformationOptions(compilerOptions, "-coarsening-factor",
                                  "-coarsening-direction");
}

//------------------------------------------------------------------------------
// Returns vw/vd.
std::pair<unsigned int, unsigned int>
getVectorizationOptions(const std::string &compilerOptions) {
  return getTransformationOptions(compilerOptions, "-vectorizing-width",
                                  "-vectorizing-direction");
}
