#ifndef PLATFORM_H
#define PLATFORM_H

#include <string>
#include <vector>

#include <CL/cl.h>

class Context;
class Device;

class Platform {
// Constructors and Destructors.
//------------------------------------------------------------------------------
public:
  Platform(unsigned int platformNumber);
  ~Platform() throw();

// Public methods.
//------------------------------------------------------------------------------
public:
  cl_platform_id getId() const;
  Context* getContext() const;
  unsigned int getDevicesNumber() const;
  const cl_device_id* getDevicesData() const ;
  Device* getDevicePointer(unsigned int index);
  const Device& getDevice(unsigned int index) const;
  const Device& getDevice(const std::string& deviceName) const;
  const Device* findDevice(const std::string& deviceName) const;

// Private Fields.
//------------------------------------------------------------------------------
private:
  cl_platform_id platformId;
  std::string platformName;
  Context* context;
  std::vector<Device*> devices;

// Private Methods.
//------------------------------------------------------------------------------
private:
// Constructors helper methods.
  void initPlatformFromNumber(unsigned int platformsNumber,
                              unsigned int platformNumer);
  void initPlatformFromName(unsigned int platformsNumber,
                            const std::string& platformName);
  void initDevicesAndContext();
  void initDevices();
  unsigned int checkPlatforms();

  unsigned int queryPlatformsNumber() const;
  cl_platform_id queryPlatformIdFromNumber(unsigned int platformsNumber,
                                           unsigned int platformNumber) const;
  cl_platform_id queryPlatformIdFromName(unsigned int platformsNumber,
                                         const std::string& platformName) const;
  void queryPlatformsId(unsigned int platformsNumber,
                        std::vector<cl_platform_id>& platformIds) const;
  unsigned int queryDevicesNumber() const;
  void queryDevicesId(cl_device_id* devicesData,
                      unsigned int devicesNumber) const;
  void buildDeviceList(cl_device_id* devicesData, unsigned int devicesNumber);

// Error management.
  void manageNoPlatformsError() const;
  void manageDeviceNotFoundError(const std::string& deviceName) const;
  void managePlatformNotFoundError(unsigned int platformNumber) const;
};

#endif
