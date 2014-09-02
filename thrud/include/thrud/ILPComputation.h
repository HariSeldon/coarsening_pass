#ifndef ILP_COMPUTATION_H
#define ILP_COMPUTATION_H

namespace llvm {
class BasicBlock;
}

using namespace llvm;

float getILP(BasicBlock *block);

#endif
