#include "Platform.h"

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <sstream>

#include "Context.h"
#include "Device.h"
#include "Utils.h"

// Constructors and Destructors.
//------------------------------------------------------------------------------
Platform::Platform(unsigned int platformNumber) {
  unsigned int platformsNumber = checkPlatforms();
  initPlatformFromNumber(platformsNumber, platformNumber);
}

Platform::~Platform() throw() { 
  devices.clear();
  delete context;
}

// Setters and Getters.
//------------------------------------------------------------------------------
cl_platform_id Platform::getId() const {
  return platformId;
}

Context* Platform::getContext() const {
  return context;
}

unsigned int Platform::getDevicesNumber() const {
  return devices.size();
}

const Device& Platform::getDevice(unsigned int index) const {
  return *devices.at(index);
}

Device* Platform::getDevicePointer(unsigned int index) {
  return devices.at(index);
}

const Device& Platform::getDevice(const std::string& deviceName) const {
  for(std::vector<Device*>::const_iterator deviceIter = devices.begin(),
                                           deviceEnd = devices.end();
                                           deviceIter != deviceEnd;
                                           ++deviceIter) {
    if((*deviceIter)->getName() == deviceName) {
      const Device* device = *deviceIter;
      return *device;
    }
  }
  manageDeviceNotFoundError(deviceName);
  return NULL;
}

const Device* Platform::findDevice(const std::string& deviceName) const {
  for(std::vector<Device*>::const_iterator deviceIter = devices.begin(),
                                           deviceEnd = devices.end();
                                           deviceIter != deviceEnd;
                                           ++deviceIter) {
    if((*deviceIter)->getName() == deviceName)
      return *deviceIter;
  }
  return NULL;
}

// Constructors helper methods.
//------------------------------------------------------------------------------
void Platform::initPlatformFromNumber(unsigned int platformsNumber, 
                                      unsigned int platformNumber) {
  platformId = queryPlatformIdFromNumber(platformsNumber, platformNumber);
  initDevicesAndContext();
}

void Platform::initDevicesAndContext() {
  initDevices();
  context = new Context(*this);
}

void Platform::initDevices() {
  unsigned int devicesNumber = queryDevicesNumber();
  cl_device_id* devicesData = new cl_device_id[devicesNumber];
  queryDevicesId(devicesData, devicesNumber);
  buildDeviceList(devicesData, devicesNumber);
  delete [] devicesData;
}

inline unsigned int Platform::checkPlatforms() {
  unsigned int platformsNumber = queryPlatformsNumber();
  if(platformsNumber == 0)
    manageNoPlatformsError();
  return platformsNumber;
}

//------------------------------------------------------------------------------
unsigned int Platform::queryPlatformsNumber() const {
  cl_uint platformNumber = 0;
  cl_int errorCode = clGetPlatformIDs(0, NULL, &platformNumber);
  verifyOutputCode(errorCode, "Cannot query the number of platforms");
  return (unsigned int) platformNumber;
}

cl_platform_id Platform::queryPlatformIdFromNumber(
                         unsigned int platformsNumber,
                         unsigned int platformNumber) const {
  if(platformsNumber <= platformNumber)
    managePlatformNotFoundError(platformNumber);
  cl_platform_id* platformsId = new cl_platform_id[platformsNumber];
  cl_int errorCode = clGetPlatformIDs(platformsNumber, platformsId, NULL);
  verifyOutputCode(errorCode, "Cannot instantiate the platform");
  cl_platform_id platformId = platformsId[platformNumber]; 
  delete [] platformsId;
  return platformId;
}

void Platform::queryPlatformsId(
               unsigned int platformsNumber,
               std::vector<cl_platform_id>& platformIds) const {
  cl_platform_id* platformsId = new cl_platform_id[platformsNumber];
  cl_int errorCode = clGetPlatformIDs(platformsNumber, platformsId, NULL);
  verifyOutputCode(errorCode, "Cannot instantiate the platforms");
  platformIds.assign(platformsId, platformsId + platformsNumber);
}

void Platform::queryDevicesId(cl_device_id* devicesData,
                              unsigned int devicesNumber) const {
  cl_int errorCode = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL,
                                    devicesNumber, devicesData, NULL);
  verifyOutputCode(errorCode, "Cannot query the available devices");
}

void Platform::buildDeviceList(cl_device_id* devicesData, 
                               unsigned int devicesNumber) {
  devices.clear();
  for(unsigned int deviceIndex = 0; deviceIndex < devicesNumber; ++deviceIndex)
    devices.push_back(new Device(devicesData[deviceIndex]));
}

unsigned int Platform::queryDevicesNumber() const {
  cl_uint deviceNumber;
  cl_int errorCode = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, 
                                    0, NULL, &deviceNumber);
  verifyOutputCode(errorCode, "Cannot query the device number");
  return (unsigned int) deviceNumber;
}

// The destruction of the pointer is demanded to the caller.
const cl_device_id* Platform::getDevicesData() const {
  cl_device_id* devicesData = new cl_device_id[devices.size()];
  for(unsigned int index = 0; index < devices.size(); ++index)
    devicesData[index] = devices.at(index)->getId();
  return devicesData;
}

// Error management.
//------------------------------------------------------------------------------
inline void Platform::manageNoPlatformsError() const {
  throw std::runtime_error("No OpenCL platforms found");
}

inline void Platform::manageDeviceNotFoundError(
                      const std::string& deviceName) const {
  throw std::runtime_error("Cannot find device: " + deviceName);
}

inline void Platform::managePlatformNotFoundError(
                      unsigned int platformNumber) const{
  std::stringstream messageStream;
  messageStream << "Cannot find platform number: " << platformNumber;
  throw std::runtime_error(messageStream.str());
}
