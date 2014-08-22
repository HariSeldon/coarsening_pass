#include "Context.h"

#include "Platform.h"
#include "Utils.h"

Context::Context(const Platform& platform) {
  cl_int errorCode;
  const cl_device_id* devicesData = platform.getDevicesData();
  context = clCreateContext(NULL, 
                            platform.getDevicesNumber(),
                            devicesData, 
                            NULL, NULL, &errorCode);
  delete [] devicesData;
  verifyOutputCode(errorCode, "Cannot create the context");
}

Context::~Context() throw() {
  clReleaseContext(context);
}

cl_context Context::getId() const {
  return context;
}
