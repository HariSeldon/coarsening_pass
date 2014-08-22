#include "bench_support.h"

#include "string.h"
#include "stdio.h"

//-----------------------------------------------------------------------------
char* getEnvString(const char* name) {
  char* envString = getenv(name);
  if(envString == NULL) return NULL;
  size_t envStringSize = strlen(envString);
  char* result = (char*) malloc((envStringSize + 1) * sizeof(char));
  memset(result, 0, envStringSize * sizeof(char));
  strncpy(result, envString, envStringSize);
  result[envStringSize] = '\0';
  return result;
}

//-----------------------------------------------------------------------------
size_t getSize(char* list, unsigned int position, unsigned int length) {
  char* string = (char*) malloc((length + 1) * sizeof(char));
  memset(string, 0, (length + 1)* sizeof(char));
  strncpy(string, list + position, length);
  string[length] = '\0';
  return (size_t) atoi(string);
}

//-----------------------------------------------------------------------------
void parseList(char* list, size_t* size1, size_t* size2) {
  size_t listSize = strlen(list);
  // Find the comma.
  unsigned int commaPosition = 0;
  while(list[commaPosition] != ',') commaPosition++;
  *size1 = getSize(list, 0, commaPosition);
  *size2 = getSize(list, commaPosition + 1, listSize - commaPosition - 1);
}

//-----------------------------------------------------------------------------
size_t* getNewSize(const char* suffix, const char* kernelName, 
                   size_t dimensions) {
  unsigned int suffixLength = strlen(suffix);
  // Read the environment variable: kernelName_suffix.
  size_t kernelNameSize = strlen(kernelName);
  size_t envVarSize = kernelNameSize + suffixLength;
  char varName[envVarSize];

  memset(varName, 0, envVarSize);
  strncpy(varName, kernelName, kernelNameSize);
  strncat(varName, suffix, suffixLength);

  char* envVarValue = getEnvString(varName);

  if(envVarValue == NULL) {
    free(envVarValue);
    return NULL;
  }

  size_t* newSize = (size_t*) malloc(dimensions * sizeof(size_t));
  memset(newSize, 0, dimensions * sizeof(size_t));

  if(dimensions == 1)
    newSize[0] = (size_t) atoi(envVarValue);
  else if(dimensions == 2)
    parseList(envVarValue, newSize, newSize + 1);

  free(envVarValue);
  return newSize; 
}

//-----------------------------------------------------------------------------
void setNewSize(size_t* origSize, size_t* newSize, const char* suffix,
                const char* kernelName, size_t dimensions) {
  size_t* envSize = getNewSize(suffix, kernelName, dimensions);
  if(envSize == NULL)
    memcpy(newSize, origSize, dimensions * sizeof(size_t));
  else
    memcpy(newSize, envSize, dimensions * sizeof(size_t));
}

//-----------------------------------------------------------------------------
void getNewSizes(size_t* origGlobalSize, size_t* origLocalSize,
                 size_t* newGlobalSize, size_t* newLocalSize,
                 const char* kernelName, size_t dimensions) {
  if(dimensions != 1 && dimensions != 2) {
    fprintf(stderr, "Wrong dimension value\n");
    exit(1);
  }

  if(origGlobalSize != NULL && newGlobalSize != NULL)
    setNewSize(origGlobalSize, newGlobalSize, "_GS", kernelName, dimensions);
  if(origLocalSize != NULL && newLocalSize != NULL)
    setNewSize(origLocalSize, newLocalSize, "_LS", kernelName, dimensions);
}

//-----------------------------------------------------------------------------
void getPlatformDevice(int* platform, int* device) {
  char* platformString = getEnvString("OCL_PLATFORM");
  char* deviceString = getEnvString("OCL_DEVICE");
  if(platformString == NULL) *platform = 0;
  else *platform = atoi(platformString);
  if(deviceString == NULL) *device = 0;
  else *device = atoi(deviceString);
}
