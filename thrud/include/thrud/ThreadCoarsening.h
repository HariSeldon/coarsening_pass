#ifndef THREAD_COARSENING_H
#define THREAD_COARSENING_H

#include "thrud/DataTypes.h"
#include "thrud/DivergenceAnalysis.h"
#include "thrud/DivergentRegion.h"

#include "llvm/Pass.h"

#include "llvm/Analysis/PostDominators.h"

using namespace llvm;

namespace llvm {
class BasicBlock;
}

class ThreadCoarsening : public FunctionPass {
public:
  enum DivRegionOption {
    FullReplication,
    TrueBranchMerging,
    FalseBranchMerging,
    FullMerging
  };

public:
  static char ID;
  ThreadCoarsening();

  virtual bool runOnFunction(Function &F);
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;

private:
  void init();

  // NDRange scaling.
  void scaleNDRange();
  void scaleSizes();
  void scaleIds();

  // Coarsening.
  void coarsenFunction();
  void replicateInst(Instruction *inst);
  void updatePlaceholderMap(Instruction *inst, InstVector &coarsenedInsts);

  void replicateRegion(DivergentRegion *region);
  void replicateRegionClassic(DivergentRegion *region);

  void initAliveMap(DivergentRegion *region, CoarseningMap &aliveMap);
  void replicateRegionImpl(DivergentRegion *region, CoarseningMap &aliveMap);
  void updateAliveMap(CoarseningMap &aliveMap, Map &regionMap);
  void updatePlaceholdersWithAlive(CoarseningMap &aliveMap);

  void replicateRegionFalseMerging(DivergentRegion *region);
  void replicateRegionTrueMerging(DivergentRegion *region);
  void replicateRegionMerging(DivergentRegion *region, unsigned int branch);
  void replicateRegionFullMerging(DivergentRegion *region);
  void applyCoarseningMap(DivergentRegion &region, unsigned int index);
  void applyCoarseningMap(BasicBlock *block, unsigned int index);
  void applyCoarseningMap(Instruction *inst, unsigned int index);
  Instruction *getCoarsenedInstruction(Instruction *inst,
                                       unsigned int coarseningIndex);

  // Manage placeholders.
  void replacePlaceholders();

  // Region merging methods.
  BasicBlock *getExitingSubregion();

  BasicBlock *createTopBranch(DivergentRegion *region);
  DivergentRegion *createCascadingFirstRegion(DivergentRegion *region,
                                              BasicBlock *pred,
                                              unsigned int branchIndex,
                                              Map &valueMap);
  Instruction *insertBooleanReduction(Instruction *base, InstVector &insts,
                                      llvm::Instruction::BinaryOps binOp);
  void updateExitPhiNodes(BasicBlock *target,
                          BasicBlock *mergedSubregionExiting,
                          InstVector &aliveFromMerged, Map &cloningMap,
                          BasicBlock *replicatedExiting,
                          CoarseningMap &aliveMap);
  void removeRedundantBlocks(DivergentRegion *region, unsigned int branchIndex);
  void buildPhiMap(DivergentRegion *region, InstVector &aliveFromMerged,
                   BasicBlock *mergedSubregionExiting, Map &result);

private:
  unsigned int direction;
  unsigned int factor;
  unsigned int stride;
  DivRegionOption divRegionOption;

  PostDominatorTree *pdt;
  DominatorTree *dt;
  SingleDimDivAnalysis *sdda;
  LoopInfo *loopInfo;
  NDRange *ndr;

  CoarseningMap cMap;
  CoarseningMap phMap;
  Map phReplacementMap;
};

#endif
