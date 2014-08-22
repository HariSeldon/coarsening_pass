#include "stdlib.h"

char* getEnvString(const char* name);
size_t getSize(char* list, unsigned int position, unsigned int length);
void parseList(char* list, size_t* size1, size_t* size2);
size_t* getNewLocalSize(size_t* origGlobalSize, size_t* origLocalSize,
                        size_t dim, const char* kernelName);
void getNewSizes(size_t* origGlobalSize, size_t* origLocalSize,
                 size_t* newGlobalSize, size_t* newLocalSize,
                 const char* kernelName, size_t dimensions);
void getPlatformDevice(int* platform, int* device);
