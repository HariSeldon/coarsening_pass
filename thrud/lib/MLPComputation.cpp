#include "thrud/MLPComputation.h"

#include "thrud/DataTypes.h"
#include "thrud/MathUtils.h"
#include "thrud/Utils.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Function.h"

#include "llvm/Analysis/PostDominators.h"

#include <algorithm>

//------------------------------------------------------------------------------
InstVector filterUsers(InstVector &insts, BasicBlock *block) {
  InstVector result;
  result.reserve(insts.size());
  std::copy_if(
      insts.begin(), insts.end(), std::back_inserter(result),
      [block](Instruction *inst) { return (inst->getParent() == block); });

  return result;
}

//------------------------------------------------------------------------------
bool isLoad(const llvm::Instruction &inst) { return isa<LoadInst>(inst); }

//------------------------------------------------------------------------------
int countLoadsBounded(Instruction *def, Instruction *user) {
  BasicBlock::iterator iter(def), end(user);
  ++iter;
  return std::count_if(iter, end, isLoad);
}

//------------------------------------------------------------------------------
int countLoads(BlockVector blocks, BasicBlock *defBlock, BasicBlock *userBlock,
               Instruction *def, Instruction *user) {

  if (defBlock == userBlock) {
    return countLoadsBounded(def, user);
  }

  int result = 0;
  for (auto block : blocks) {
    if (block == defBlock) {
      result += countLoadsBounded(def, block->end());
      continue;
    }
    if (block == userBlock) {
      result += countLoadsBounded(block->begin(), user);
      continue;
    }

    result += countLoadsBounded(block->begin(), block->end());
  }
  return result;
}

//------------------------------------------------------------------------------
// FIXME: this might not work with loops.
BlockVector getRegionBlocks(BasicBlock *defBlock, BasicBlock *userBlock) {
  BlockVector result;
  BlockStack stack;
  stack.push(userBlock);

  while (!stack.empty()) {
    // Pop the first block.
    BasicBlock *block = stack.top();
    result.push_back(block);
    stack.pop();

    // Don't put to the stack the defBlock predecessors.
    if (block == defBlock)
      continue;

    // Push to the stack the defBlock predecessors.
    for (pred_iterator iter = pred_begin(block), end = pred_end(block);
         iter != end; ++iter) {
      BasicBlock *pred = *iter;
      stack.push(pred);
    }
  }

  return result;
}

//------------------------------------------------------------------------------
int computeDistance(Instruction *def, Instruction *user) {
  BasicBlock *defBlock = def->getParent();
  BasicBlock *userBlock = user->getParent();

  // Manage the special case in which the user is a phi-node.
  if (PHINode *phi = dyn_cast<PHINode>(user)) {
    for (unsigned int index = 0; index < phi->getNumIncomingValues(); ++index) {
      if (def == phi->getIncomingValue(index)) {
        userBlock = phi->getIncomingBlock(index);
        BlockVector blocks = getRegionBlocks(defBlock, userBlock);
        return countLoads(blocks, defBlock, userBlock, def, userBlock->end());
      }
    }
  }

  BlockVector blocks = getRegionBlocks(defBlock, userBlock);
  return countLoads(blocks, defBlock, userBlock, def, user);
}

//------------------------------------------------------------------------------
// MLP computation.
// MLP: count the number of loads that fall in each load-use interval
// (interval between a load and the first use of the loaded value).
float getMLP(BasicBlock *block) {
  std::vector<int> distances;

  for (auto inst = block->begin(), end = block->end(); inst != end; ++inst) {
    if (isa<LoadInst>(inst)) {
      InstVector users = findUsers(inst);
      users = filterUsers(users, block);

      std::transform(
          users.begin(), users.end(), distances.begin(),
          [inst](Instruction *user) { return computeDistance(inst, user); });
    }
  }

  return getAverage(distances);
}
