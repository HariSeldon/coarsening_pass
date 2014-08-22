#ifndef EVENT_H
#define EVENT_H

#include <CL/cl.h>

class Context;

class Event {

// Constructors and Destructors.
//------------------------------------------------------------------------------
public:
  Event();
  ~Event() throw();

// Public methods.
//------------------------------------------------------------------------------
public:
  cl_event* getId() const;
  unsigned long int computeDuration() const;

// Private Fields.
//------------------------------------------------------------------------------
private:
  cl_event* event;
};

#endif
