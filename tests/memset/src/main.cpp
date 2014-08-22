#include <algorithm>
#include <cassert>
#include <iostream>
#include <sstream>
#include <string.h>
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
constexpr int SIZE = 1024;
const float CONSTANT = 3.14f;
std::vector<size_t> globalWorkSize = { SIZE, SIZE };
std::vector<size_t> localWorkSize = { 16, 16 };

const std::string kernelFileName = "memset.cl";
std::string kernelName = "memset2D";

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
std::vector<float> inputHost;

// Device data.
Buffer *input = nullptr;

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
  Program program(context, kernelFile.c_str());
  Queue queue(*context, device, Queue::EnableProfiling);
  enqueWriteCommands(queue);

  if (!program.build(device, "-DCONSTANT=" + std::to_string(CONSTANT))) {
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
void initialization(int , char **) {
  assert(globalWorkSize.size() == localWorkSize.size() &&
         "Mismatching local and global work sizes");
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
}

//-----------------------------------------------------------------------------
void hostMemoryAlloc() {
  width = new cl_int(globalWorkSize[0]);
  int bufferSize = std::accumulate(globalWorkSize.begin(), globalWorkSize.end(),
                                   1, std::multiplies<int>());

  std::random_device randomDevice;
  std::mt19937_64 gen(randomDevice());
  std::uniform_real_distribution<float> distribution;

  inputHost.assign(bufferSize, 0.f);
  std::generate_n(inputHost.begin(), bufferSize, [&] {
    return (distribution(gen));
  });
}

//-----------------------------------------------------------------------------
void deviceMemoryAlloc() {
  input = new Buffer(*(platform->getContext()), Buffer::ReadWrite,
                     inputHost.size() * sizeof(float), nullptr);
}

//-----------------------------------------------------------------------------
void enqueWriteCommands(Queue &queue) {
  queue.writeBuffer(*input, inputHost.size() * sizeof(float),
                    (void *)inputHost.data());
  queue.finish();
}

//-----------------------------------------------------------------------------
void enqueReadCommands(Queue &queue) {
  queue.readBuffer(*input, inputHost.size() * sizeof(float),
                   (void *)inputHost.data());
  queue.finish();
}

//-----------------------------------------------------------------------------
void setKernelArguments() {
  kernel->setArgument(0, *input);
  kernel->setArgument(1, sizeof(cl_int), (void *)width);
}

//-----------------------------------------------------------------------------
void run(Queue &queue) {
  queue.run(*kernel, globalWorkSize.size(), 0, globalWorkSize.data(),
            localWorkSize.data());
  queue.finish();
}

//-----------------------------------------------------------------------------
void verify() {
  std::vector<float> proof(inputHost.size(), CONSTANT);

  assert(memcmp(proof.data(), inputHost.data(),
                inputHost.size() * sizeof(float)) == 0);
}
