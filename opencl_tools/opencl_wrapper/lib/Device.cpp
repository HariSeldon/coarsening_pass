#include "Device.h"

#include "Utils.h"

Device::Device(cl_device_id device) : device(device) {
  deviceName = queryName();
}

Device::~Device() throw() {}

cl_device_id Device::getId() const { return device; }

size_t Device::queryMaxWorkGroupSize() const {
  return DeviceInfoTraits<size_t>::getDeviceInfo(device,
                                                 CL_DEVICE_MAX_WORK_GROUP_SIZE);
}

unsigned long int Device::queryLocalMemorySize() const {
  return DeviceInfoTraits<unsigned long int>::getDeviceInfo(
      device, CL_DEVICE_LOCAL_MEM_SIZE);
}

const std::string &Device::getName() const { return deviceName; }

std::string Device::queryName() const {
  return DeviceInfoTraits<std::string>::getDeviceInfo(device, CL_DEVICE_NAME);
}

//------------------------------------------------------------------------------
template <typename returnType>
returnType
DeviceInfoTraits<returnType>::getDeviceInfo(cl_device_id deviceId,
                                            cl_device_info deviceInfoName) {
  returnType result;
  cl_int errorCode = clGetDeviceInfo(deviceId, deviceInfoName,
                                     sizeof(returnType), &result, NULL);
  verifyOutputCode(errorCode, "Error querying device info: ");
  return result;
}

size_t getDeviceInfoSize(cl_device_id deviceId, cl_device_info deviceInfoName) {
  size_t result;
  cl_int errorCode =
      clGetDeviceInfo(deviceId, deviceInfoName, 0, NULL, &result);
  verifyOutputCode(errorCode, "Error querying device info size: ");
  return result;
}

std::string
DeviceInfoTraits<std::string>::getDeviceInfo(cl_device_id deviceId,
                                             cl_device_info deviceInfoName) {
  size_t resultSize = getDeviceInfoSize(deviceId, deviceInfoName);
  char *rawResult = new char[resultSize];
  cl_int errorCode =
      clGetDeviceInfo(deviceId, deviceInfoName, resultSize, rawResult, NULL);
  verifyOutputCode(errorCode, "Error querying device info: ");
  std::string result(rawResult, resultSize);
  delete[] rawResult;
  return result;
}
