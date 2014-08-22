#ifndef UTILS_H
#define UTILS_H

#include "thrud/DataTypes.h"
#include "thrud/DivergentRegion.h"
#include "thrud/RegionBounds.h"

#include "llvm/IR/Argument.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/ValueMap.h"

#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"

#include "llvm/Transforms/Utils/ValueMapper.h"

#include <deque>
#include <set>
#include <vector>

using namespace llvm;

extern const char *BARRIER;

// Loop management.
bool isInLoop(const Instruction &inst, LoopInfo *loopInfo);
bool isInLoop(const Instruction *inst, LoopInfo *loopInfo);
bool isInLoop(const BasicBlock *block, LoopInfo *loopInfo);

// OpenCL management.
bool isKernel(const Function *function);

void safeIncrement(std::map<std::string, int> &inputMap, std::string key);

// Map management.
// Apply the given map to the given instruction.
void applyMap(Instruction *Inst, Map &map);
void applyMap(BasicBlock *block, Map &map);
void applyMapToPHIs(BasicBlock *block, Map &map);
void applyMapToPhiBlocks(PHINode *Phi, Map &map);
void applyMap(Instruction *Inst, CoarseningMap &map, unsigned int CF);
void applyMap(InstVector &insts, Map &map, InstVector &result);

void renameValueWithFactor(Value *value, StringRef oldName, unsigned int index);

// Prints to stderr the given map. For debug only.
void dump(const Map &map);
void dumpV2V(const V2VMap &map);
void dumpCoarseningMap(const CoarseningMap &map);

// Replate all the usages of O with N.
void replaceUses(Value *O, Value *N);
void remapBlocksInPHIs(BasicBlock *Target, BasicBlock *OldBlock,
                       BasicBlock *NewBlock);
void InitializeMap(Map &map, const InstVector &TIds, const InstVector &NewTIds,
                   unsigned int CI, unsigned int CF);

// Instruction management.
BranchVector FindBranches(Function &F);
template <class InstructionType>
std::vector<InstructionType *> getInsts(Function &F);
unsigned int GetOperandPosition(User *U, Value *value);
PhiVector getPHIs(BasicBlock *block);

// Container management.
// Check if the given element is present in the given container.
template <class T>
bool isPresent(const T *value, const std::vector<T *> &vector);
template <class T>
bool isPresent(const T *value, const std::vector<const T *> &vector);
template <class T> bool isPresent(const T *value, const std::set<T *> &vector);
template <class T>
bool isPresent(const T *value, const std::set<const T *> &vector);
template <class T> bool isPresent(const T *value, const std::deque<T *> &deque);

bool isPresent(const Instruction *inst, const BlockVector &value);
bool isPresent(const Instruction *inst, std::vector<BlockVector *> &value);

BasicBlock *findImmediatePostDom(BasicBlock *block,
                                 const PostDominatorTree *pdt);

// Block management.
void changeBlockTarget(BasicBlock *block, BasicBlock *newTarget,
                       unsigned int branchIndex = 0);

// Region analysis.
BlockVector InsertChildren(BasicBlock *block, BlockSet &Set);

// Cloning support.
void CloneDominatorInfo(BasicBlock *block, Map &map, DominatorTree *dt);

// Domination.
BranchInst *findOutermostBranch(BranchSet &blocks, const DominatorTree *dt);

bool isDominated(const Instruction *inst, BranchSet &blocks,
                 const DominatorTree *dt);
bool isDominated(const Instruction *inst, BranchVector &blocks,
                 const DominatorTree *dt);
bool isDominated(const BasicBlock *block, const BlockVector &blocks,
                 const DominatorTree *dt);
bool dominates(const BasicBlock *block, const BranchVector &blocks,
               const DominatorTree *dt);
bool dominatesAll(const BasicBlock *block, const BlockVector &blocks,
                  const DominatorTree *dt);
bool postdominatesAll(const BasicBlock *block, const BlockVector &blocks,
                      const PostDominatorTree *pdt);

ValueVector ToValueVector(InstVector &Insts);

template <class type> void dumpSet(const std::set<type *> &toDump);
template <class type> void dumpVector(const std::vector<type *> &toDump);
void dumpIntVector(const std::vector<int> &toDump);

// Divergence Utils.

Function *getOpenCLFunctionByName(std::string calleeName, Function *caller);

//------------------------------------------------------------------------------
bool isBarrier(Instruction *inst);
bool isMathFunction(Instruction *inst);
bool isMathName(std::string fName);
bool isLocalMemoryAccess(Instruction *inst);
bool isLocalMemoryStore(Instruction *inst);
bool isLocalMemoryLoad(Instruction *inst);
bool isIntCast(Instruction *inst);

//------------------------------------------------------------------------------
bool isUsedOutsideOfDefiningBlock(const Instruction *inst);
Instruction *findFirstUser(Instruction *inst);
Instruction *findLastUser(Instruction *inst);
InstVector findUsers(llvm::Value *value);

#endif
