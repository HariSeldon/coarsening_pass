#ifndef PROGRAM_H
#define PROGRAM_H

#include <string>
#include <vector>

#include <CL/cl.h>

class Context;
class Device;
class Kernel;

class Program {
// Constructors and Destructors.
//------------------------------------------------------------------------------
public:
  Program(Context* context, const std::string& sourceFile);
  Program(Context* context, const Device& device, const std::string& binaryFile);
  ~Program() throw();

// Public methods.
//------------------------------------------------------------------------------
public:
  cl_program getId() const;
  Kernel* createKernel(const char* name);

  bool build(const Device& device) const;
  bool build(const Device& device, const std::string& options) const;

  std::string getBuildLog(const Device& device) const;

  std::string getBinary(const Device& device) const;
  std::string getSourceCode() const;
  // OCL 1.2 only.
  //std::string getKernelsList() const;

// Private Fields.
//------------------------------------------------------------------------------
private:
  cl_program program;
  Context* context;
  bool createdFromSource;

// Private Methods.
//------------------------------------------------------------------------------
private:
  void createFromSource(const std::string& filePath);
  void createFromBinary(const Device& device, const std::string& filePath);

  bool build(cl_device_id deviceId, const char* options) const;

  void forceRecompilation() const;

  size_t getBuildLogSize(const Device& device) const;
  std::string getBuildLogText(const Device& device, size_t buildLogSize) const;

  size_t getSourceCodeSize() const;
  std::string getSourceCodeText(size_t sourceCodeSize) const;

  // OCL 1.2 only.
  //size_t getKernelsNumber() const;
  //size_t getKernelsListSize() const;

  unsigned int getDeviceIndex(const Device& device) const;
  std::vector<cl_device_id> queryDevices() const;
  cl_uint queryDevicesNumber() const;
  void queryDevicesId(cl_device_id* devicesData, cl_uint devicesNumber) const;
  size_t getBinarySize(const Device& device) const;
  std::vector<size_t> getBinariesSize(unsigned int devicesNumber) const;
  std::string getBinaryText(unsigned int deviceIndex, 
                            const std::vector<size_t>& binariesSize) const;
};

#endif
