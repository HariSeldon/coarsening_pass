#include "Event.h"

#include <stdexcept>

#include <CL/cl.h>

#include "Context.h"
#include "Utils.h"

Event::Event() : event(new cl_event()) { 
}

Event::~Event() throw() {
  clReleaseEvent(*event);
  delete(event);
}

cl_event* Event::getId() const {
  return event;
}

unsigned long int Event::computeDuration() const {
  if(event == NULL)
    throw std::runtime_error("Error computing event duration. \
                              Event is not initialized");
  cl_int errorCode;
  cl_ulong start, end;
  errorCode = clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_END,
                                      sizeof(cl_ulong), &end, NULL);
  verifyOutputCode(errorCode, "Error querying the event end time");
  errorCode = clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_START,
                                      sizeof(cl_ulong), &start, NULL);
  verifyOutputCode(errorCode, "Error querying the event start time");
  return static_cast<unsigned long>(end - start);
}
