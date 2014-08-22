#include "thrud/OCLEnv.h"

#include "llvm/IR/Type.h"

#include "llvm/IR/Function.h"

const int OCLEnv::BANK_NUMBER = 32;
const int OCLEnv::BANK_WIDTH = 4;
const int OCLEnv::WARP_SIZE = 32;
const int OCLEnv::CACHELINE_SIZE = 128;
const int OCLEnv::UNKNOWN_MEMORY_LOCATION = -1;
const unsigned int OCLEnv::LOCAL_AS = 3;

OCLEnv::OCLEnv(Function &function, const NDRange *ndRange, const NDRangeSpace &ndRangeSpace)
    : ndRange(ndRange), ndRangeSpace(ndRangeSpace) {
  setup(function);
}

void OCLEnv::setup(Function &function) {
  // Go through the function arguements and setup the map.
  for (Function::arg_iterator iter = function.arg_begin(),
                              iterEnd = function.arg_end();
       iter != iterEnd; ++iter) {
    llvm::Value *argument = iter;
    llvm::Type *type = argument->getType();
    // Only set the value of the argument if it is an integer.
    if (type->isIntegerTy()) {
      // FIXME! Read the input configuration from a file.
      argumentMap.insert(std::pair<llvm::Value *, int>(argument, 1024));
    }
  }
}

const NDRange *OCLEnv::getNDRange() const { return ndRange; }

const NDRangeSpace &OCLEnv::getNDRangeSpace() const { return ndRangeSpace; }

int OCLEnv::resolveValue(llvm::Value *value) const {
  std::map<llvm::Value *, int>::const_iterator iter = argumentMap.find(value);
  assert(iter != argumentMap.end() && "Argument is not in argument map!");
  return iter->second;
}
