#include <CL/cl.h>

#define PERCENT_DIFF_ERROR_THRESHOLD 0.0001

void cl_initialization(cl_device_id &deviceId, cl_context &clContext,
                       cl_command_queue &queue);
void verifyOutputCode(cl_int valueToCheck, const char* errorMessage);
bool isError(cl_int valueToCheck);

