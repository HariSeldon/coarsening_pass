#ifndef DEVICE_H
#define DEVICE_H

#include <string>

#include <CL/cl.h>

class Context;

class Device {
public:
  enum DeviceType {
    CPUDevice = CL_DEVICE_TYPE_CPU,
    GPUDevice =  CL_DEVICE_TYPE_GPU,
    AcceleratorDevice = CL_DEVICE_TYPE_ACCELERATOR,
    DefaultDevice = CL_DEVICE_TYPE_DEFAULT,
    AllDevice = CL_DEVICE_TYPE_ALL
  };  

// Constructors and Destructors.
//------------------------------------------------------------------------------
public:
  Device(cl_device_id device);
  ~Device() throw();

// Public Methods.
//------------------------------------------------------------------------------
public:
  cl_device_id getId() const;
  const std::string& getName() const;
  size_t queryMaxWorkGroupSize() const;
  unsigned long int queryLocalMemorySize() const;  

// Private Methods.
//------------------------------------------------------------------------------
private:
  std::string queryName() const;

// Private Fields.
//------------------------------------------------------------------------------
private:
  cl_device_id device;
  std::string deviceName;
};

// Traits.
//------------------------------------------------------------------------------
template <typename returnType> struct DeviceInfoTraits {
  static returnType getDeviceInfo(cl_device_id deviceId, 
                                  cl_device_info deviceInfoName);
};

template <> struct DeviceInfoTraits<std::string> {
  static std::string getDeviceInfo(cl_device_id deviceId, 
                                   cl_device_info deviceInfoName);
};

#endif
