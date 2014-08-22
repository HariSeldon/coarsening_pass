#include "Utils.h"

#include <iostream>
#include <stdexcept>

#include "bench_support.h"

void throwException(cl_int errorCode, const char* errorMessage);
std::string getErrorDescription(cl_uint errorCode);

// -----------------------------------------------------------------------------
void cl_initialization(cl_device_id &deviceId, cl_context &context,
                       cl_command_queue &queue) {

  cl_int errorCode = 0;
  cl_uint platformsNumber = 0;
  cl_uint devicesNumber = 0;
  int platformNumber = 0;
  int deviceNumber = 0;

  // Get Ids.
  getPlatformDevice(&platformNumber, &deviceNumber);

  // Get number of platforms.
  errorCode = clGetPlatformIDs(0, NULL, &platformsNumber);
  verifyOutputCode(errorCode, "Cannot query the number of platforms");

  // Get platform.
  cl_platform_id* platformsId = new cl_platform_id[platformsNumber];
  errorCode = clGetPlatformIDs(platformsNumber, platformsId, NULL);
  verifyOutputCode(errorCode, "Cannot instantiate the platform");
  cl_platform_id platformId = platformsId[platformNumber];
  delete [] platformsId;

  // Get number of devices.
  errorCode =
      clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, 0, NULL, &devicesNumber);
  verifyOutputCode(errorCode, "Cannot query the device number");

  // Get device.
  cl_device_id* deviceIds = new cl_device_id[devicesNumber];
  errorCode = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, devicesNumber,
                             deviceIds, NULL);
  verifyOutputCode(errorCode, "Cannot instantiate the device");
  deviceId = deviceIds[deviceNumber]; 
  delete [] deviceIds;

  // Get device name.
  size_t deviceNameSize;
  errorCode =
      clGetDeviceInfo(deviceId, CL_DEVICE_NAME, 0, NULL, &deviceNameSize);
  char* deviceName = new char[deviceNameSize];
  errorCode = clGetDeviceInfo(deviceId, CL_DEVICE_NAME, deviceNameSize,
                              deviceName, NULL);
  std::cout << "Running on: " << deviceName << "\n";

  // Create OpenCL Context.
  context = clCreateContext(NULL, 1, &deviceId, NULL, NULL, &errorCode);
  verifyOutputCode(errorCode, "Cannot create the context");

  //Create Command Queue.
  queue = clCreateCommandQueue(context, deviceId, 0, &errorCode);
  verifyOutputCode(errorCode, "Error creating the queue");
}

// -----------------------------------------------------------------------------
void verifyOutputCode(cl_int valueToCheck, const char* errorMessage) {
  if (isError(valueToCheck))
    throwException(valueToCheck, errorMessage);
}

// -----------------------------------------------------------------------------
inline bool isError(cl_int valueToCheck) {
  return valueToCheck != CL_SUCCESS;
}

// -----------------------------------------------------------------------------
inline void throwException(cl_int errorCode, const char* errorMessage) {
  std::string errorString(errorMessage);
  errorString += ": " + getErrorDescription(errorCode);
  throw std::runtime_error(errorString);
}

// -----------------------------------------------------------------------------
std::string getErrorDescription(cl_uint errorCode) {
  switch(errorCode) {
    case CL_SUCCESS:
      return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:
      return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:
      return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:
      return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:
      return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:
      return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
      return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP:
      return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH:
      return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
      return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE:
      return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE:
      return "CL_MAP_FAILURE";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:
      return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
      return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case CL_INVALID_VALUE:
      return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:
      return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:
      return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:
      return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:
      return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES:
      return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE:
      return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR:
      return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT:
      return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
      return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE:
      return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER:
      return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY:
      return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:
      return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:
      return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:
      return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:
      return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION:
      return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL:
      return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:
      return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:
      return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:
      return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS:
      return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:
      return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:
      return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:
      return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET:
      return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST:
      return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT:
      return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION:
      return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT:
      return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE:
      return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL:
      return "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_GLOBAL_WORK_SIZE:
      return "CL_INVALID_GLOBAL_WORK_SIZE";
    default:
      return "UNKNOWN ERROR CODE";
  }
}
