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
constexpr int SIZE = 512;
std::vector<size_t> globalWorkSize = { SIZE };
std::vector<size_t> localWorkSize = { 128 };

const std::string kernelFileName = "mv.cl";
std::string kernelName = "";

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
std::vector<float> inputMatrixHost;
std::vector<float> inputVectorHost;
std::vector<float> outputVectorHost;

// Device data.
Buffer *inputMatrix = nullptr;
Buffer *inputVector = nullptr;
Buffer *outputVector = nullptr;

cl_int *width = nullptr;
cl_int *height = nullptr;

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
  Program program(context, kernelFile);
  Queue queue(*context, device, Queue::EnableProfiling);
  enqueWriteCommands(queue);

  if (!program.build(device)) {
    std::cout << "Error building the program: \n";
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
void initialization(int argNumber, char **arguments) {
  assert(globalWorkSize.size() == localWorkSize.size() &&
         "Mismatching local and global work sizes");

  if (argNumber != 2) {
    std::cerr << "Must specify kernel name\n";
    exit(1);
  }

  kernelName = std::string(arguments[1]);
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
  width = new cl_int(globalWorkSize[0]);
  height = new cl_int(globalWorkSize[0]);

  std::random_device randomDevice;
  std::mt19937_64 gen(randomDevice());
  std::uniform_real_distribution<float> distribution;

  inputVectorHost.assign(*width, 0.f);
  inputMatrixHost.assign(*width * (*height), 0.f);
  outputVectorHost.assign(*height, 0.f);

  std::generate_n(inputVectorHost.begin(), *width,
                  [&] { return (distribution(gen)); });
  std::generate_n(inputMatrixHost.begin(), *width * (*height),
                  [&] { return (distribution(gen)); });
}

//-----------------------------------------------------------------------------
void deviceMemoryAlloc() {
  inputVector = new Buffer(*(platform->getContext()), Buffer::ReadOnly,
                           inputVectorHost.size() * sizeof(float), nullptr);
  inputMatrix = new Buffer(*(platform->getContext()), Buffer::ReadOnly,
                               inputMatrixHost.size() * sizeof(float), nullptr);
  outputVector = new Buffer(*(platform->getContext()), Buffer::WriteOnly,
                      outputVectorHost.size() * sizeof(float), nullptr);
}

//-----------------------------------------------------------------------------
void enqueWriteCommands(Queue &queue) {
  queue.writeBuffer(*inputVector, inputVectorHost.size() * sizeof(float),
                    (void *)inputVectorHost.data());
  queue.writeBuffer(*inputMatrix, inputMatrixHost.size() * sizeof(float), 
                    (void *)inputMatrixHost.data()); 
  queue.finish();
}

//-----------------------------------------------------------------------------
void enqueReadCommands(Queue &queue) {
  queue.readBuffer(*outputVector, outputVectorHost.size() * sizeof(float),
                   (void *)outputVectorHost.data());
  queue.finish();
}

//-----------------------------------------------------------------------------
void setKernelArguments() {
  kernel->setArgument(0, *inputMatrix);
  kernel->setArgument(1, *inputVector);
  kernel->setArgument(2, sizeof(cl_uint), (void *)width);
  kernel->setArgument(3, sizeof(cl_uint), (void *)height);
  kernel->setArgument(4, *outputVector);
  if(kernelName == "MatVecMulCoalesced0") {
    kernel->setArgument(5, localWorkSize[0] * sizeof(float), nullptr);
  }
}

//-----------------------------------------------------------------------------
void run(Queue &queue) {
  queue.run(*kernel, globalWorkSize.size(), 0, globalWorkSize.data(),
            localWorkSize.data());
  queue.finish();
}

//-----------------------------------------------------------------------------
void verify() {
  for (int row = 0; row < 32; ++row) {
    float result = 0.0f;
    for (int column = 0; column < *width; ++column) {
      result += inputMatrixHost[row * (*width) + column] * inputVectorHost[column];
    }
    std::cout << outputVectorHost[row] << " " << result << "\n";
//    if (abs(outputVectorHost[row] - result) >= 0.001f) {
//      std::cout << "Error\n";
//      exit(1);
//    }
  }
}
