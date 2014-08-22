#include <algorithm>
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <stdio.h>

#include "Buffer.h"
#include "Device.h"
#include "Event.h"
#include "Kernel.h"
#include "Platform.h"
#include "Program.h"
#include "Queue.h"
#include "SystemConfiguration.h"

#include "bench_support.h"

//-----------------------------------------------------------------------------
constexpr int SIZE = 256;
std::vector<size_t> globalWorkSize = { SIZE, SIZE };
std::vector<size_t> localWorkSize = { 16, 16 };

const std::string kernelFileName = "mm.cl";
std::string kernelName = "mm";

//-----------------------------------------------------------------------------
void initialization(int argNumber, char **arguments);
void hostMemoryAlloc();
void deviceMemoryAlloc();
void setKernelArguments();
void enqueWriteCommands(Queue &queue);
void enqueReadCommands(Queue &queue);
void run(Queue &queue);
void freeMemory();
void setNDRangeSizes();
void verify();

//-----------------------------------------------------------------------------
// Runtime components.
Platform *platform;
Kernel *kernel;

// Host data.
std::vector<float> aHost;
std::vector<float> bHost;
std::vector<float> outputHost;

// Device data.
Buffer *matrixA;
Buffer *matrixB;
Buffer *output;

cl_int *height = nullptr;
cl_int *width = nullptr;

// Device.
int PLATFORM_ID = 0;
int DEVICE_ID = 0;

//-----------------------------------------------------------------------------
int main(int argNumber, char **arguments) {
  initialization(argNumber, arguments);

  getPlatformDevice(&PLATFORM_ID, &DEVICE_ID);

  platform = new Platform(PLATFORM_ID);
  Context *context = platform->getContext();
  Device device = platform->getDevice(DEVICE_ID);
  setNDRangeSizes();
  std::cout << "Device name: " << device.getName() << "\n";
  hostMemoryAlloc();
  deviceMemoryAlloc();
  std::string kernelFile = KERNEL_DIR + kernelFileName;
  Program program(context, std::string(kernelFile.c_str()));
  Queue queue(*context, device, Queue::EnableProfiling);
  enqueWriteCommands(queue);

  if (!program.build(device)) {
    std::cout << "Error building the program: "
              << "\n";
    std::cout << program.getBuildLog(device) << "\n";
    return 1;
  }

  kernel = program.createKernel(kernelName.c_str());
  setKernelArguments();
  run(queue);
  enqueReadCommands(queue);
  verify();
  freeMemory();
  return 0;
}

//-----------------------------------------------------------------------------
void initialization(int, char **) {
  assert(globalWorkSize.size() == localWorkSize.size() &&
         "Mismatching local and global work sizes");

  kernelName = "mm";
}

//-----------------------------------------------------------------------------
void setNDRangeSizes() {
  std::vector<size_t> newGlobalWorkSize(globalWorkSize.size(), 0);
  std::vector<size_t> newLocalWorkSize(localWorkSize.size(), 0);
  getNewSizes(globalWorkSize.data(), localWorkSize.data(),
              newGlobalWorkSize.data(), newLocalWorkSize.data(),
              kernelName.c_str(), globalWorkSize.size());

  globalWorkSize.clear();
  localWorkSize.clear();

  std::copy(newGlobalWorkSize.begin(), newGlobalWorkSize.end(),
            std::back_inserter(globalWorkSize));
  std::copy(newLocalWorkSize.begin(), newLocalWorkSize.end(),
            std::back_inserter(localWorkSize));
}

//-----------------------------------------------------------------------------
void freeMemory() {
  delete kernel;
  delete platform;
  delete width;
  delete height;
}

//-----------------------------------------------------------------------------
void hostMemoryAlloc() {
  int bufferSize = std::accumulate(globalWorkSize.begin(), globalWorkSize.end(),
                                   1, std::multiplies<int>());

  std::random_device randomDevice;
  std::mt19937_64 gen(randomDevice());
  std::uniform_real_distribution<float> distribution;

  aHost.assign(bufferSize, 0.f);
  bHost.assign(bufferSize, 0.f);
  std::generate_n(aHost.begin(), bufferSize, [&] {
    return (distribution(gen));
  });
  std::generate_n(bHost.begin(), bufferSize, [&] {
    return (distribution(gen));
  });
  outputHost.assign(bufferSize, 0.f);
  width = new cl_int(globalWorkSize[0]);
  height = new cl_int(globalWorkSize[1]);
}

//-----------------------------------------------------------------------------
void deviceMemoryAlloc() {
  matrixA = new Buffer(*(platform->getContext()), Buffer::ReadOnly,
                     aHost.size() * sizeof(float), nullptr);
  matrixB = new Buffer(*(platform->getContext()), Buffer::ReadOnly,
                     bHost.size() * sizeof(float), nullptr);
  output = new Buffer(*(platform->getContext()), Buffer::WriteOnly,
                      outputHost.size() * sizeof(float), nullptr);
}

//-----------------------------------------------------------------------------
void enqueWriteCommands(Queue &queue) {
  queue.writeBuffer(*matrixA, aHost.size() * sizeof(float),
                    (void *)aHost.data());
  queue.finish();
  queue.writeBuffer(*matrixB, bHost.size() * sizeof(float),
                    (void *)bHost.data());
  queue.finish();
}

//-----------------------------------------------------------------------------
void enqueReadCommands(Queue &queue) {
  queue.readBuffer(*output, outputHost.size() * sizeof(float),
                   (void *)outputHost.data());
  queue.finish();
}

//-----------------------------------------------------------------------------
void setKernelArguments() {
  kernel->setArgument(0, *matrixA);
  kernel->setArgument(1, *matrixB);
  kernel->setArgument(2, *output);
  kernel->setArgument(3, sizeof(cl_int), (void *)width);
  kernel->setArgument(4, sizeof(cl_int), (void *)height);
}

//-----------------------------------------------------------------------------
void run(Queue &queue) {
  queue.run(*kernel, globalWorkSize.size(), 0, globalWorkSize.data(),
            localWorkSize.data());
  queue.finish();
}

//-----------------------------------------------------------------------------
void verify() {
  float* cpuHostC = new float [SIZE * SIZE];
  for(unsigned int row = 0; row < SIZE; ++row) {
    for(unsigned int column = 0; column < SIZE; ++column) {
      cpuHostC[row * SIZE + column] = 0.0f;
      for(unsigned int index = 0; index < SIZE; ++index) 
        cpuHostC[row * SIZE + column] += aHost[row * SIZE + index] * 
                                         bHost[index * SIZE + column];

      if(abs(outputHost[row * SIZE + column] - cpuHostC[row * SIZE + column]) 
         >= 0.001f) {
        std::cout << "Error in the computation:";
        exit(1);
      }
    }
  }

  delete [] cpuHostC;
}
