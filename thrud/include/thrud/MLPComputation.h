#ifndef MLP_COMPUTATION
#define MLP_COMPUTATION

#include "llvm/Analysis/PostDominators.h"

namespace llvm {
class BasicBlock;
}

using namespace llvm;

float getMLP(BasicBlock *block);

#endif
