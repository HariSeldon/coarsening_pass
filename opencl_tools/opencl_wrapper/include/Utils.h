#ifndef UTILS_H
#define UTILS_H

#include <string>

#include <CL/cl.h>

void verifyOutputCode(cl_int valueToCheck, const char* errorMessage);
bool isError(cl_int valueToCheck);
void throwException(cl_int errorCode, const char* errorMessage);

std::string getErrorDescription(cl_uint errorCode);

#endif
