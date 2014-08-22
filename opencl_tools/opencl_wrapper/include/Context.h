#ifndef CONTEXT_H
#define CONTEXT_H

#include <CL/cl.h>

class Platform;

class Context {

// Constructors and Destructors.
//------------------------------------------------------------------------------
public:
  Context(const Platform& platform);
  ~Context() throw();

// Public methods.
//-----------------------------------------------------------------------------
public:
   cl_context getId() const;

// Private Fields.
//------------------------------------------------------------------------------
private:
  cl_context context;
};

#endif
