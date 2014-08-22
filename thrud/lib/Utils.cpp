#include "thrud/Utils.h"

#include "thrud/DivergentRegion.h"
#include "thrud/RegionBounds.h"
#include "thrud/OCLEnv.h"

#include "llvm/ADT/STLExtras.h"

#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/PostDominators.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"

#include "llvm/Support/raw_ostream.h"

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <algorithm>

// OpenCL function names.
const char *BARRIER = "barrier";

//------------------------------------------------------------------------------
bool isInLoop(const Instruction &inst, LoopInfo *loopInfo) {
  const BasicBlock *block = inst.getParent();
  return loopInfo->getLoopFor(block) != nullptr;
}

bool isInLoop(const Instruction *inst, LoopInfo *loopInfo) {
  const BasicBlock *block = inst->getParent();
  return loopInfo->getLoopFor(block) != nullptr;
}

//------------------------------------------------------------------------------
bool isInLoop(const BasicBlock *block, LoopInfo *loopInfo) {
  return loopInfo->getLoopFor(block) != nullptr;
}

//------------------------------------------------------------------------------
bool isKernel(const Function *function) {
  const Module *module = function->getParent();
  const llvm::NamedMDNode *kernelsMD =
      module->getNamedMetadata("opencl.kernels");

  if (!kernelsMD)
    return false;

  for (int index = 0, end = kernelsMD->getNumOperands(); index != end;
       ++index) {
    const llvm::MDNode &kernelMD = *kernelsMD->getOperand(index);
    if (kernelMD.getOperand(0) == function)
      return true;
  }

  return false;
}

//------------------------------------------------------------------------------
void applyMapToPhiBlocks(PHINode *Phi, Map &map) {
  for (unsigned int index = 0; index < Phi->getNumIncomingValues(); ++index) {
    BasicBlock *OldBlock = Phi->getIncomingBlock(index);
    Map::const_iterator It = map.find(OldBlock);

    if (It != map.end()) {
      // I am not proud of this.
      BasicBlock *NewBlock =
          const_cast<BasicBlock *>(cast<BasicBlock>(It->second));
      Phi->setIncomingBlock(index, NewBlock);
    }
  }
}

//------------------------------------------------------------------------------
void applyMap(Instruction *Inst, CoarseningMap &map, unsigned int CF) {
  for (unsigned op = 0, opE = Inst->getNumOperands(); op != opE; ++op) {
    Instruction *Op = dyn_cast<Instruction>(Inst->getOperand(op));
    CoarseningMap::iterator It = map.find(Op);

    if (It != map.end()) {
      InstVector &instVector = It->second;
      Value *NewValue = instVector.at(CF);
      Inst->setOperand(op, NewValue);
    }
  }
}

//------------------------------------------------------------------------------
void applyMap(Instruction *Inst, Map &map) {
  for (unsigned op = 0, opE = Inst->getNumOperands(); op != opE; ++op) {
    Value *Op = Inst->getOperand(op);

    Map::const_iterator It = map.find(Op);
    if (It != map.end())
      Inst->setOperand(op, It->second);
  }

  if (PHINode *Phi = dyn_cast<PHINode>(Inst))
    applyMapToPhiBlocks(Phi, map);
}

//------------------------------------------------------------------------------
void applyMapToPHIs(BasicBlock *block, Map &map) {
  for (auto phi = block->begin(); isa<PHINode>(phi); ++phi)
    applyMap(phi, map);
}

//------------------------------------------------------------------------------
void applyMap(BasicBlock *block, Map &map) {
  for (auto iter = block->begin(), end = block->end(); iter != end; ++iter)
    applyMap(iter, map);
}

//------------------------------------------------------------------------------
void applyMap(InstVector &insts, Map &map, InstVector &result) {
  result.clear();
  result.reserve(insts.size());

  for (auto inst : insts) {
    Value *newValue = map[inst];
    if (newValue != nullptr) {
      if (Instruction *newInst = dyn_cast<Instruction>(newValue)) {
        result.push_back(newInst);
      }
    }
  }
}

//------------------------------------------------------------------------------
void dump(const Map &map) {
  errs() << "==== Map ====\n";
  for (auto iter : map)
    errs() << iter.first->getName() << " -> " << iter.second->getName() << "\n";
  errs() << "=============\n";
}

//------------------------------------------------------------------------------
void dumpV2V(const V2VMap &map) {
  errs() << "==== Map ====\n";
  for (auto iter : map)
    errs() << iter.first->getName() << " -> " << iter.second->getName() << "\n";
  errs() << "=============\n";
}

//------------------------------------------------------------------------------
void dumpCoarseningMap(const CoarseningMap &cMap) {
  llvm::errs() << "------------------------------\n";
  for (auto iter : cMap) {
    const InstVector &entry = iter.second;
    const Instruction *inst = iter.first;
    llvm::errs() << "Key: ";
    inst->dump();
    llvm::errs() << " ";
    dumpVector(entry);
    llvm::errs() << "\n";
  }
  llvm::errs() << "------------------------------\n";
}

//------------------------------------------------------------------------------
void replaceUses(Value *oldValue, Value *newValue) {
  std::vector<User *> users;
  std::copy(oldValue->user_begin(), oldValue->user_end(),
            std::back_inserter(users));

  std::for_each(users.begin(), users.end(), [oldValue, newValue](User *user) {
    if (user != newValue)
      user->replaceUsesOfWith(oldValue, newValue);
  });
}

//------------------------------------------------------------------------------
BranchVector FindBranches(Function &F) {
  std::vector<BranchInst *> Result;

  for (inst_iterator inst = inst_begin(F), E = inst_end(F); inst != E; ++inst)
    if (BranchInst *branchInst = dyn_cast<BranchInst>(&*inst))
      if (branchInst->isConditional() && branchInst->getNumSuccessors() > 1)
        Result.push_back(branchInst);

  return Result;
}

//------------------------------------------------------------------------------
template <class InstructionType>
std::vector<InstructionType *> getInsts(Function &F) {
  std::vector<InstructionType *> Result;

  for (inst_iterator inst = inst_begin(F), E = inst_end(F); inst != E; ++inst)
    if (InstructionType *SpecInst = dyn_cast<InstructionType>(&*inst))
      Result.push_back(SpecInst);

  return Result;
}

//------------------------------------------------------------------------------
unsigned int GetOperandPosition(User *U, Value *value) {
  for (unsigned int index = 0; index < U->getNumOperands(); ++index)
    if (value == U->getOperand(index))
      return index;
  assert(0 && "Value not used by User");
}

//------------------------------------------------------------------------------
BasicBlock *findImmediatePostDom(BasicBlock *block,
                                 const PostDominatorTree *pdt) {
  return pdt->getNode(block)->getIDom()->getBlock();
}

//------------------------------------------------------------------------------
template <class type> void dumpSet(const std::set<type *> &toDump) {
  for (auto element : toDump)
    element->dump();
}

template <> void dumpSet(const BlockSet &toDump) {
  for (auto element : toDump)
    errs() << element->getName() << " ";
  errs() << "\n";
}

template void dumpSet(const InstSet &toDump);
template void dumpSet(const std::set<BranchInst *> &toDump);
template void dumpSet(const BlockSet &toDump);

//------------------------------------------------------------------------------
template <class type> void dumpVector(const std::vector<type *> &toDump) {
  errs() << "Size: " << toDump.size() << "\n";
  for (auto element : toDump)
    element->dump();
}

// Template specialization for BasicBlock.
template <> void dumpVector(const BlockVector &toDump) {
  errs() << "Size: " << toDump.size() << "\n";
  for (auto element : toDump)
    errs() << element->getName() << " -- ";
  errs() << "\n";
}

template void dumpVector(const std::vector<Instruction *> &toDump);
template void dumpVector(const std::vector<BranchInst *> &toDump);
template void dumpVector(const std::vector<DivergentRegion *> &toDump);
template void dumpVector(const std::vector<PHINode *> &toDump);
template void dumpVector(const std::vector<Value *> &toDump);

void dumpIntVector(const std::vector<int> &toDump) {
  for (auto element : toDump) {
    errs() << element << ", ";
  }
}

//-----------------------------------------------------------------------------
// This is black magic. Don't touch it.
void CloneDominatorInfo(BasicBlock *BB, Map &map, DominatorTree *DT) {
  assert(DT && "DominatorTree is not available");
  Map::iterator BI = map.find(BB);
  assert(BI != map.end() && "BasicBlock clone is missing");
  BasicBlock *NewBB = cast<BasicBlock>(BI->second);

  // NewBB already got dominator info.
  if (DT->getNode(NewBB))
    return;

  assert(DT->getNode(BB) && "BasicBlock does not have dominator info");
  // Entry block is not expected here. Infinite loops are not to cloned.
  assert(DT->getNode(BB)->getIDom() &&
         "BasicBlock does not have immediate dominator");
  BasicBlock *BBDom = DT->getNode(BB)->getIDom()->getBlock();

  // NewBB's dominator is either BB's dominator or BB's dominator's clone.
  BasicBlock *NewBBDom = BBDom;
  Map::iterator BBDomI = map.find(BBDom);
  if (BBDomI != map.end()) {
    NewBBDom = cast<BasicBlock>(BBDomI->second);
    if (!DT->getNode(NewBBDom))
      CloneDominatorInfo(BBDom, map, DT);
  }
  DT->addNewBlock(NewBB, NewBBDom);
}

//------------------------------------------------------------------------------
BranchInst *findOutermostBranch(BranchSet &branches, const DominatorTree *dt) {
  for (auto branch : branches)
    if (!isDominated(branch, branches, dt))
      return branch;
  return nullptr;
}

//------------------------------------------------------------------------------
bool isDominated(const Instruction *inst, BranchVector &branches,
                 const DominatorTree *dt) {
  const BasicBlock *block = inst->getParent();
  return std::any_of(branches.begin(), branches.end(),
                     [inst, block, dt](BranchInst *branch) {
    const BasicBlock *currentBlock = branch->getParent();
    return inst != branch && dt->dominates(currentBlock, block);
  });
}

//------------------------------------------------------------------------------
bool isDominated(const Instruction *inst, BranchSet &branches,
                 const DominatorTree *dt) {
  const BasicBlock *block = inst->getParent();
  return std::any_of(branches.begin(), branches.end(),
                     [inst, block, dt](BranchInst *branch) {
    const BasicBlock *currentBlock = branch->getParent();
    return inst != branch && dt->dominates(currentBlock, block);
  });
}

//------------------------------------------------------------------------------
bool isDominated(const BasicBlock *block, const BlockVector &blocks,
                 const DominatorTree *dt) {
  return std::any_of(blocks.begin(), blocks.end(),
                     [block, dt](BasicBlock *iter) {
    return block != iter && dt->dominates(iter, block);
  });
}

//------------------------------------------------------------------------------
bool dominatesAll(const BasicBlock *block, const BlockVector &blocks,
                  const DominatorTree *dt) {
  return std::all_of(
      blocks.begin(), blocks.end(),
      [block, dt](BasicBlock *iter) { return dt->dominates(block, iter); });
}

//------------------------------------------------------------------------------
bool postdominatesAll(const BasicBlock *block, const BlockVector &blocks,
                      const PostDominatorTree *pdt) {
  return std::all_of(
      blocks.begin(), blocks.end(),
      [block, pdt](BasicBlock *iter) { return pdt->dominates(block, iter); });
}

//------------------------------------------------------------------------------
void changeBlockTarget(BasicBlock *block, BasicBlock *newTarget,
                       unsigned int branchIndex) {
  TerminatorInst *terminator = block->getTerminator();
  assert(terminator->getNumSuccessors() &&
         "The target can be change only if it is unique");
  terminator->setSuccessor(branchIndex, newTarget);
}

//------------------------------------------------------------------------------
ValueVector ToValueVector(InstVector &insts) {
  ValueVector result;
  std::copy(insts.begin(), insts.end(), std::back_inserter(result));
  return result;
}

//------------------------------------------------------------------------------
PhiVector getPHIs(BasicBlock *block) {
  PhiVector result;
  PHINode *phi = nullptr;
  for (auto iter = block->begin(); (phi = dyn_cast<PHINode>(iter)); ++iter) {
    result.push_back(phi);
  }
  return result;
}

//------------------------------------------------------------------------------
void remapBlocksInPHIs(BasicBlock *block, BasicBlock *oldBlock,
                       BasicBlock *newBlock) {
  Map phiMap;
  phiMap[oldBlock] = newBlock;
  applyMapToPHIs(block, phiMap);
}

//------------------------------------------------------------------------------
Function *getOpenCLFunctionByName(std::string calleeName, Function *caller) {
  Module &module = *caller->getParent();
  Function *callee = module.getFunction(calleeName);

  if (callee == nullptr)
    return nullptr;

  assert(callee->arg_size() == 1 && "Wrong OpenCL function");
  return callee;
}

//------------------------------------------------------------------------------
// Region and Branch Analysis.
//------------------------------------------------------------------------------
bool isBarrier(Instruction *inst) {
  if (CallInst *callInst = dyn_cast<CallInst>(inst)) {
    Function *function = callInst->getCalledFunction();
    return function->getName() == "barrier";
  }
  return false;
}

//------------------------------------------------------------------------------
bool isLocalMemoryAccess(Instruction *inst) {
  return isLocalMemoryStore(inst) || isLocalMemoryLoad(inst);
}

//------------------------------------------------------------------------------
bool isLocalMemoryStore(Instruction *inst) {
  if (StoreInst *storeInst = dyn_cast<StoreInst>(inst)) {
    return (storeInst->getPointerAddressSpace() == OCLEnv::LOCAL_AS);
  }
  return false;
}

//------------------------------------------------------------------------------
bool isLocalMemoryLoad(Instruction *inst) {
  if (LoadInst *loadInst = dyn_cast<LoadInst>(inst)) {
    return (loadInst->getPointerAddressSpace() == OCLEnv::LOCAL_AS);
  }
  return false;
}

//------------------------------------------------------------------------------
bool isMathFunction(Instruction *inst) {
  if (CallInst *callInst = dyn_cast<CallInst>(inst)) {
    Function *function = callInst->getCalledFunction();
    return isMathName(function->getName().str());
  }
  return false;
}

//------------------------------------------------------------------------------
bool isMathName(std::string functionName) {
  bool begin = (functionName[0] == '_' && functionName[1] == 'Z');
  bool value = ((functionName.find("sin") != std::string::npos) ||
                (functionName.find("cos") != std::string::npos) ||
                (functionName.find("exp") != std::string::npos) ||
                (functionName.find("acos") != std::string::npos) ||
                (functionName.find("asin") != std::string::npos) ||
                (functionName.find("atan") != std::string::npos) ||
                (functionName.find("tan") != std::string::npos) ||
                (functionName.find("ceil") != std::string::npos) ||
                (functionName.find("exp2") != std::string::npos) ||
                (functionName.find("exp10") != std::string::npos) ||
                (functionName.find("fabs") != std::string::npos) ||
                (functionName.find("abs") != std::string::npos) ||
                (functionName.find("fma") != std::string::npos) ||
                (functionName.find("max") != std::string::npos) ||
                (functionName.find("fmax") != std::string::npos) ||
                (functionName.find("min") != std::string::npos) ||
                (functionName.find("fmin") != std::string::npos) ||
                (functionName.find("log") != std::string::npos) ||
                (functionName.find("log2") != std::string::npos) ||
                (functionName.find("mad") != std::string::npos) ||
                (functionName.find("pow") != std::string::npos) ||
                (functionName.find("pown") != std::string::npos) ||
                (functionName.find("root") != std::string::npos) ||
                (functionName.find("rootn") != std::string::npos) ||
                (functionName.find("sqrt") != std::string::npos) ||
                (functionName.find("trunc") != std::string::npos) ||
                (functionName.find("rsqrt") != std::string::npos) ||
                (functionName.find("rint") != std::string::npos) ||
                (functionName.find("ceil") != std::string::npos) ||
                (functionName.find("round") != std::string::npos) ||
                (functionName.find("hypot") != std::string::npos) ||
                (functionName.find("cross") != std::string::npos) ||
                (functionName.find("mix") != std::string::npos) ||
                (functionName.find("clamp") != std::string::npos) ||
                (functionName.find("normalize") != std::string::npos) ||
                (functionName.find("floor") != std::string::npos));
  return begin && value;
}

//------------------------------------------------------------------------------
void safeIncrement(std::map<std::string, int> &map, std::string key) {
  std::map<std::string, int>::iterator iter = map.find(key);
  if (iter == map.end())
    map[key] = 1;
  else
    map[key] += 1;
}

//------------------------------------------------------------------------------
bool isUsedOutsideOfDefiningBlock(const Instruction *inst) {
  if (inst->use_empty())
    return false;
  if (isa<PHINode>(inst))
    return true;
  const BasicBlock *block = inst->getParent();
  for (auto userIter = inst->user_begin(), userEnd = inst->user_end();
       userIter != userEnd; ++userIter) {
    const User *user = *userIter;
    if (cast<Instruction>(user)->getParent() != block || isa<PHINode>(user))
      return true;
  }
  return false;
}

//------------------------------------------------------------------------------
// Build a vector with all the uses of the given value.
InstVector findUsers(llvm::Value *value) {
  InstVector result;
  for (auto user = value->user_begin(), end = value->user_end(); user != end;
       ++user) {
    if (Instruction *inst = dyn_cast<Instruction>(*user)) {
      result.push_back(inst);
    }
  }
  return result;
}

//------------------------------------------------------------------------------
InstVector filterUsers(llvm::Instruction *used, InstVector &users) {
  BasicBlock *block = used->getParent();
  InstVector result;
  std::copy_if(
      users.begin(), users.end(), std::back_inserter(result),
      [block](Instruction *inst) { return inst->getParent() == block; });
  return result;
}

//------------------------------------------------------------------------------
// Find the last user of input instruction in its parent block.
// Return nullptr if no use is found.
Instruction *findLastUser(Instruction *inst) {
  InstVector users = findUsers(inst);
  users = filterUsers(inst, users);
  Instruction *lastUser = nullptr;
  int maxDistance = 0;

  BasicBlock::iterator begin(inst);
  for (auto inst : users) {
    BasicBlock::iterator blockIter(inst);
    int currentDist = std::distance(begin, blockIter);
    if (currentDist > maxDistance) {
      maxDistance = currentDist;
      lastUser = inst;
    }
  }

  return lastUser;
}

//------------------------------------------------------------------------------
// Find first user of input instruction in its parent block.
// Return nullptr if no use is found.
Instruction *findFirstUser(Instruction *inst) {
  InstVector users = findUsers(inst);
  Instruction *firstUser = nullptr;
  int minDistance = inst->getParent()->size();

  BasicBlock::iterator begin(inst);
  for (auto inst : users) {
    BasicBlock::iterator blockIter(inst);
    int currentDist = std::distance(begin, blockIter);
    if (currentDist < minDistance) {
      minDistance = currentDist;
      firstUser = inst;
    }
  }

  return firstUser;
}

//------------------------------------------------------------------------------
bool isIntCast(Instruction *inst) {
  if (CallInst *call = dyn_cast<CallInst>(inst)) {
    Function *callee = call->getCalledFunction();
    std::string name = callee->getName();
    bool begin = (name[0] == '_' && name[1] == 'Z');
    bool value = ((name.find("as_uint") != std::string::npos) ||
                  (name.find("as_int") != std::string::npos));
    return begin && value;
  }
  return false;
}

//------------------------------------------------------------------------------
void renameValueWithFactor(Value *value, StringRef oldName,
                           unsigned int index) {
  if (!oldName.empty())
    value->setName(oldName + "..cf" + Twine(index + 2));
}

// isPresent.
//------------------------------------------------------------------------------
template <class T>
bool isPresent(const T *value, const std::vector<T *> &values) {
  auto result = std::find(values.begin(), values.end(), value);
  return result != values.end();
}

template bool isPresent(const Instruction *value, const InstVector &values);
template bool isPresent(const Value *value, const ValueVector &values);

template <class T>
bool isPresent(const T *value, const std::vector<const T *> &values) {
  auto result = std::find(values.begin(), values.end(), value);
  return result != values.end();
}

template bool isPresent(const Instruction *value,
                        const ConstInstVector &values);
template bool isPresent(const Value *value, const ConstValueVector &values);

template <class T> bool isPresent(const T *value, const std::set<T *> &values) {
  auto result = std::find(values.begin(), values.end(), value);
  return result != values.end();
}

template bool isPresent(const Instruction *value, const InstSet &values);

template <class T>
bool isPresent(const T *value, const std::set<const T *> &values) {
  auto result = std::find(values.begin(), values.end(), value);
  return result != values.end();
}

template bool isPresent(const Instruction *value, const ConstInstSet &values);

template <class T> bool isPresent(const T *value, const std::deque<T *> &d) {
  auto result = std::find(d.begin(), d.end(), value);
  return result != d.end();
}

template bool isPresent(const BasicBlock *value, const BlockDeque &deque);

bool isPresent(const Instruction *inst, const BlockVector &value) {
  const BasicBlock *BB = inst->getParent();
  return isPresent<BasicBlock>(BB, value);
}

bool isPresent(const Instruction *inst, std::vector<BlockVector *> &value) {
  for (auto Iter = value.begin(), E = value.end(); Iter != E; ++Iter)
    if (isPresent(inst, **Iter))
      return true;
  return false;
}
