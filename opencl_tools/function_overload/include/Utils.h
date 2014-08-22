#include "CL/cl.h"

#include <string>
#include <utility>

#define OCL_BLOCK_SIZE_X "OCL_BLOCK_SIZE_X"
#define OCL_BLOCK_SIZE_Y "OCL_BLOCK_SIZE_Y"
#define OCL_COMPILER "OCL_COMPILER"
#define OCL_COMPILER_OPTIONS "OCL_COMPILER_OPTIONS"
#define OCL_REPETITIONS "OCL_REPETITIONS"
#define CLC_DIRECTORY "/home/s1158370/src/libclc/"
#define OCL_INPUT_FILE "/tmp/ocl_input.cl"
#define OCL_OUTPUT_FILE "/tmp/ocl_output.cl"
#define PTX_FILE "/tmp/tmp.ptx"
#define BC_FILE "/tmp/bc.ll"

//------------------------------------------------------------------------------
// Runtime function prototypes.

#define CL_CREATE_PROGRAM_WITH_SOURCE_NAME "clCreateProgramWithSource"
typedef cl_program (*clCreateProgramWithSourceFunction)
  (cl_context, cl_uint, const char**, const size_t*, cl_int*);

#define CL_BUILD_PROGRAM_NAME "clBuildProgram"
typedef cl_int (*clBuildProgramFunction)
  (cl_program, cl_uint, const cl_device_id *, const char *,
   void (*)(cl_program, void *), void *);

#define CL_CREATE_PROGRAM_WITH_BINARY_NAME "clCreateProgramWithBinary"
typedef cl_program (*clCreateProgramWithBinaryFunction)
  (cl_context, cl_uint, const cl_device_id *, const size_t *,
   const unsigned char **, cl_int *, cl_int *);

#define CL_ENQUEUE_NDRANGE_KERNEL_NAME "clEnqueueNDRangeKernel"
typedef cl_int (*clEnqueueNDRangeKernelFunction)
  (cl_command_queue command_queue,
   cl_kernel kernel,
   cl_uint work_dim,
   const size_t* global_work_offset,
   const size_t* global_work_size,
   const size_t* local_work_size,
   cl_uint num_events_in_wait_list,
   const cl_event* event_wait_list,
   cl_event* event);

#define CL_CREATE_KERNEL_NAME "clCreateKernel"
typedef cl_kernel (*clCreateKernelFunction)
  (cl_program,
   const char*,
   cl_int*);

#define CL_RELEASE_PROGRAM_NAME "clReleaseProgram"
typedef cl_int (*clReleaseProgramFunction)(cl_program);

#define CL_CREATE_PROGRAM_WITH_SOURCE_NAME "clCreateProgramWithSource"
typedef cl_program (*clCreateProgramWithSourceFunction)
  (cl_context, cl_uint, const char **, const size_t *, cl_int *);

#define CL_BUILD_PROGRAM_NAME "clBuildProgram"
typedef cl_int (*clBuildProgramFunction)
  (cl_program, cl_uint, const cl_device_id *, const char *, 
   void (*)(cl_program, void *), void *);

#define CL_RETAIN_PROGRAM_NAME "clRetainProgram"
typedef cl_int (*clRetainProgramFunction)
  (cl_program);

#define CL_GET_PROGRAM_INFO_NAME "clGetProgramInfo"
typedef cl_int (*clGetProgramInfoFunction)
  (cl_program,
   cl_program_info,
   size_t,
   void *,
   size_t *);

#define CL_GET_PROGRAM_BUILD_INFO_NAME "clGetProgramBuildInfo"
typedef cl_int (*clGetProgramBuildInfoFunction)
  (cl_program,
   cl_device_id,
   cl_program_build_info,
   size_t,
   void *,
   size_t *);

#define CL_CREATE_PROGRAM_WITH_BINARY_NAME "clCreateProgramWithBinary"
typedef cl_program (*clCreateProgramWithBinaryFunction)
  (cl_context,
   cl_uint,
   const cl_device_id*,
   const size_t*,
   const unsigned char**,
   cl_int*,
   cl_int*);

#define CL_CREATE_COMMAND_QUEUE_NAME "clCreateCommandQueue"
typedef cl_command_queue (*clCreateCommandQueueFunction)
  (cl_context,
   cl_device_id,
   cl_command_queue_properties,
   cl_int*);


//------------------------------------------------------------------------------
// Support functions.

char* readFile(const char* filePath, size_t* size);
void writeFile(std::string filePath, const std::string &data);
std::string getEnvString(const char* name, const char* defValue="");

char* getProgramSourceCode(cl_program program, size_t* codeSize);
cl_device_id* getProgramDevices(cl_program program,
                                unsigned int* devicesNumber);
cl_context getProgramContext(cl_program program);
cl_context getKernelContext(cl_kernel kernel);
cl_device_id getDeviceFromContext(cl_context context, unsigned int deviceNumber);
std::string getKernelName(cl_kernel kernel);

void verifyOutputCode(cl_int valueToCheck, const char* errorMessage);
bool isError(cl_int valueToCheck);

std::string getMangledFileName(const char *fileName, int seed);
cl_int dumpError(cl_int errorCode);

unsigned long int computeEventDuration(cl_event* event);

bool computeNDRangeDim(unsigned int dimensions,
                       const size_t* globalSize, const size_t* localSize,
                       size_t* newGlobalSize, size_t *newLocalSize);

void enqueueKernel(cl_command_queue command_queue,
                   cl_kernel kernel,
                   cl_uint work_dim,
                   const size_t* global_work_offset,
                   const size_t* global_work_size,
                   const size_t* local_work_size,
                   cl_uint num_events_in_wait_list,
                   const cl_event* event_wait_list,
                   cl_event* event,
                   unsigned int repetitions,
                   const std::string &kernelName);

void enqueueKernelSingleThread(cl_command_queue command_queue,
                               cl_kernel kernel,
                               cl_uint work_dim,
                               const size_t* global_work_offset,
                               const size_t* global_work_size,
                               const size_t* local_work_size,
                               cl_uint num_events_in_wait_list,
                               const cl_event* event_wait_list,
                               cl_event* event,
                               unsigned int repetitions,
                               const std::string &kernelName);

void enqueueKernelNoTime(cl_command_queue command_queue,
                         cl_kernel kernel,
                         cl_uint work_dim,
                         const size_t* global_work_offset,
                         const size_t* global_work_size,
                         const size_t* local_work_size,
                         cl_uint num_events_in_wait_list,
                         const cl_event* event_wait_list,
                         cl_event* event,
                         unsigned int repetitions,
                         const std::string &kernelName);

//------------------------------------------------------------------------------
// Compiler / OpenCL functions.

int compileWithAxtor(std::string &inputFile, 
                     std::string &clangOptions, std::string &optOptions, 
                     std::string &outputFile,
                     int seed);

std::string buildPTXCommandLine(std::string &inputFile,
                                std::string &compilerOptions,
                                std::string &outputFile);

void splitCompilerOptions(std::string& clangOptions, std::string& oclOptions);

std::pair<unsigned int,unsigned int>
getCoarseningOptions(const std::string& compilerOptions);
std::pair<unsigned int,unsigned int>
getVectorizationOptions(const std::string& compilerOptions);
