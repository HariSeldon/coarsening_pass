#include "Program.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "Context.h"
#include "Device.h"
#include "FileUtils.h"
#include "Kernel.h"
#include "Utils.h"

#include <algorithm>
#include <iterator>

// Constructors and Destructors.
//------------------------------------------------------------------------------
Program::Program(Context* context, const std::string& sourceFile) : 
                 context(context) {
  createFromSource(sourceFile); 
}

//------------------------------------------------------------------------------
Program::Program(Context* context, const Device& device, 
                 const std::string& binaryFile) : 
                 context(context) {
  createFromBinary(device, binaryFile); 
}

//------------------------------------------------------------------------------
Program::~Program() throw () {
  clReleaseProgram(program);
}

//------------------------------------------------------------------------------
void Program::createFromSource(const std::string& sourceFile) {
  std::string sourceString = readFile(sourceFile);
  sourceString += '\0';
  const char* sourceData = sourceString.data();
  size_t sourceSize = sourceString.length();

  cl_int errorCode;
  program = clCreateProgramWithSource(context->getId(), 1, 
                                      (const char**) &sourceData,
                                      &sourceSize, &errorCode);
  verifyOutputCode(errorCode, "Error creating the program with source");
  createdFromSource = true;
}

//------------------------------------------------------------------------------
void Program::createFromBinary(const Device& device, 
                               const std::string& binaryFile) {
  std::string sourceString = readFile(binaryFile);
  sourceString[sourceString.size()] = '\0';

  cl_device_id deviceId = device.getId();
  size_t binarySize = sourceString.length();
  const char* sourceData = sourceString.data();

  cl_int binaryStatus;
  cl_int errorCode;
  program = clCreateProgramWithBinary(context->getId(), 1,
                                      &deviceId,
                                      &binarySize,
                                      (const unsigned char**) &sourceData,
                                      &binaryStatus,
                                      &errorCode);
  verifyOutputCode(errorCode, "Error creating the program from binary");
  verifyOutputCode(binaryStatus, "Invalid binary");
  createdFromSource = false;
}

//------------------------------------------------------------------------------
cl_program Program::getId() const {
  return program;
}

//------------------------------------------------------------------------------
Kernel* Program::createKernel(const char* name) {
  return new Kernel(*this, name);
}

//------------------------------------------------------------------------------
bool Program::build(const Device& device) const {
  return build(device.getId(), "");
}

//------------------------------------------------------------------------------
bool Program::build(const Device& device, const std::string& options) const {
  return build(device.getId(), options.c_str());
}

//------------------------------------------------------------------------------
bool Program::build(cl_device_id deviceId, const char* options) const {
  cl_int errorCode = clBuildProgram(program, 1, &deviceId, options, NULL, NULL);
  return !isError(errorCode);
}

// Get build Log.
//------------------------------------------------------------------------------
std::string Program::getBuildLog(const Device& device) const {
  size_t buildLogSize = getBuildLogSize(device);
  return getBuildLogText(device, buildLogSize);
}

//------------------------------------------------------------------------------
size_t Program::getBuildLogSize(const Device& device) const {
  size_t buildLogSize;
  cl_int errorCode = clGetProgramBuildInfo(program, device.getId(), 
                                           CL_PROGRAM_BUILD_LOG, 0, NULL,
                                           &buildLogSize);
  verifyOutputCode(errorCode, "Error querying the build log size");
  return buildLogSize;
}

//------------------------------------------------------------------------------
std::string Program::getBuildLogText(const Device& device, 
                                     size_t buildLogSize) const {
  char* buildLog = new char[buildLogSize+1];
  cl_int errorCode = clGetProgramBuildInfo(program, device.getId(),
                                           CL_PROGRAM_BUILD_LOG, buildLogSize,
                                           buildLog, NULL);
  verifyOutputCode(errorCode, "Error querying the build log");
  std::string buildLogString(buildLog);
  delete [] buildLog;
  return buildLogString;
}

// Get program source.
//------------------------------------------------------------------------------
std::string Program::getSourceCode() const {
  if(!createdFromSource)
    throw std::runtime_error("Cannot query the source code when program \
                              created from binary");
  size_t sourceCodeSize = getSourceCodeSize();
  return getSourceCodeText(sourceCodeSize);
}

//------------------------------------------------------------------------------
size_t Program::getSourceCodeSize() const {
  size_t sourceCodeSize;
  cl_int errorCode = clGetProgramInfo(program, CL_PROGRAM_SOURCE, 0, NULL, 
                                      &sourceCodeSize);
  verifyOutputCode(errorCode, "Error querying the source code size");
  return sourceCodeSize;
}

//------------------------------------------------------------------------------
std::string Program::getSourceCodeText(size_t sourceCodeSize) const {
  char* sourceCode = new char[sourceCodeSize];
  cl_int errorCode = clGetProgramInfo(program, CL_PROGRAM_SOURCE, 
                                      sourceCodeSize, sourceCode, NULL);
  verifyOutputCode(errorCode, "Error querying the source code");
  return std::string(sourceCode);
}

// Get program binary.
//------------------------------------------------------------------------------
std::string Program::getBinary(const Device& device) const {
  cl_uint devicesNumber = queryDevicesNumber();
  std::vector<size_t> binariesSize = getBinariesSize(devicesNumber);
  unsigned int deviceIndex = getDeviceIndex(device);
  return getBinaryText(deviceIndex, binariesSize);
}

//------------------------------------------------------------------------------
unsigned int Program::getDeviceIndex(const Device& device) const {
  std::vector<cl_device_id> devices = queryDevices();
  std::vector<cl_device_id>::iterator deviceIter = std::find(devices.begin(), 
                                                             devices.end(), 
                                                             device.getId());
  if(deviceIter == devices.end())
    std::runtime_error("Requested binary for device not associated with the \
                        current program");
  unsigned int deviceIndex = std::distance(devices.begin(), deviceIter);
  return deviceIndex;
}

//------------------------------------------------------------------------------
std::vector<cl_device_id> Program::queryDevices() const {
  cl_uint devicesNumber = queryDevicesNumber(); 
  cl_device_id* devicesData = new cl_device_id[devicesNumber]; 
  queryDevicesId(devicesData, devicesNumber);
  std::vector<cl_device_id> result(devicesData, devicesData + devicesNumber);
  delete [] devicesData;
  return result;
}

//------------------------------------------------------------------------------
unsigned int Program::queryDevicesNumber() const {
  cl_uint devicesNumber;
  cl_int errorCode = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, 
                                      sizeof(cl_uint), &devicesNumber, NULL);
  verifyOutputCode(errorCode, "Error querying the number of devices \
                               associated to the progam");
  return devicesNumber;
}

//------------------------------------------------------------------------------
void Program::queryDevicesId(cl_device_id* devicesData, 
                             cl_uint devicesNumber) const {
  cl_int errorCode = clGetProgramInfo(program, CL_PROGRAM_DEVICES,
                                      sizeof(cl_device_id) * devicesNumber,
                                      devicesData, NULL);
  verifyOutputCode(errorCode, "Error querying the devices associated \
                               to the program");
}

//------------------------------------------------------------------------------
std::vector<size_t> Program::getBinariesSize(unsigned int devicesNumber) const {
  size_t* binarySizes = new size_t[devicesNumber];
  cl_int errorCode = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                                      devicesNumber * sizeof(size_t),
                                      binarySizes, NULL);
  verifyOutputCode(errorCode, "Error getting the program binary size: ");
  std::vector<size_t> result(binarySizes, binarySizes + devicesNumber);
  delete [] binarySizes;
  return result;
}

//------------------------------------------------------------------------------
// FIXME. Split in multiple functions.
std::string Program::getBinaryText(
                     unsigned int deviceIndex,
                     const std::vector<size_t>& binariesSize) const {
  unsigned int binariesNumber = binariesSize.size();
  char** binaryPrograms = new char* [binariesNumber];
  for (unsigned int index = 0; index < binariesNumber; ++index) { 
    binaryPrograms[index] = NULL;
  }
  binaryPrograms[deviceIndex] = new char[binariesSize[deviceIndex] + 1];

  cl_int errorCode = clGetProgramInfo(program, CL_PROGRAM_BINARIES,
                                      0, binaryPrograms, NULL);
  verifyOutputCode(errorCode, "Error getting the program binary: ");

  binaryPrograms[deviceIndex][binariesSize[deviceIndex]] = '\0';
  std::string binaryProgramString(binaryPrograms[deviceIndex]);
  delete [] binaryPrograms[deviceIndex];
  delete [] binaryPrograms;
  return binaryProgramString;
}

// Get kernels list. Only in OCL 1.2
//------------------------------------------------------------------------------
//std::string Program::getKernelsList() const {
//  size_t kernelsNumber = getKernelsNumber();
//  size_t listSize = getKernelsListSize();
//  char* kernelsList = new char[listSize];
//  cl_int errorCode = clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES,
//                                      listSize * sizeof(char),
//                                      &kernelsList, NULL); 
//  std::string result(kernelsList);
//  delete [] kernelsList;
//  return result;
//}
//
//size_t Program::getKernelsNumber() const {
//  size_t kernelsNumber;
//  cl_int errorCode = clGetProgramInfo(program, CL_PROGRAM_NUM_KERNELS,
//                                      sizeof(size_t),
//                                      &kernelsNumber, NULL);
//  verifyOutputCode(errorCode, "Error getting the kernels number: ");
//  return kernelsNumber;
//}
//
//size_t Program::getKernelsListSize() const {
//  size_t listSize;
//  cl_int errorCode = clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES,
//                                      sizeof(size_t),
//                                      NULL, &listSize);
//  verifyOutputCode(errorCode, "Error getting the kernels list size: ");
//  return listSize;
//}
