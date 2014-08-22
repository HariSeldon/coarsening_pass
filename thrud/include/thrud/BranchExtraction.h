#ifndef BRANCH_EXTRACTION_H
#define BRANCH_EXTRACTION_H

#include "llvm/Pass.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"

using namespace llvm;

class DivergentRegion;
class SingleDimDivAnalysis;

class BranchExtraction : public FunctionPass {
public:
  static char ID;
  BranchExtraction();

  virtual bool runOnFunction(Function &function);
  virtual void getAnalysisUsage(AnalysisUsage &au) const;

private:
  void extractBranches(DivergentRegion *region);
  void isolateRegion(DivergentRegion *region);

private:
  LoopInfo *loopInfo;
  DominatorTree *dt;
  PostDominatorTree *pdt;
  SingleDimDivAnalysis *sdda;
};

#endif
