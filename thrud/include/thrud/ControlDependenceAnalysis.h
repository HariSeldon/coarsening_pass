#ifndef CONTROL_DEPENCE_ANALYSIS_H
#define CONTROL_DEPENCE_ANALYSIS_H

// Implementation based on Ferrante et al. "Program Dependecene Graph and Its
// Use in Optimization". page 324
// The notation of the variables is taken from the paper.

#include "thrud/DataTypes.h"

#include "llvm/Pass.h"

#include "llvm/Analysis/PostDominators.h"

#include <map>
#include <vector>

using namespace llvm;

class ControlDependenceAnalysis : public FunctionPass {
public:
  static char ID;
  ControlDependenceAnalysis();
  ~ControlDependenceAnalysis();

  virtual bool runOnFunction(Function &function);
  virtual void getAnalysisUsage(AnalysisUsage &au) const;

  bool dependsOn(BasicBlock *first, BasicBlock *second);
  bool dependsOn(Instruction *first, Instruction *second);
  bool dependsOnAny(BasicBlock *block, BlockVector &blocks);
  bool dependsOnAny(Instruction *inst, InstVector &insts);
  bool controls(BasicBlock *first, BasicBlock *second);

  void dump();

private:
  void findS(Function &function);
  void findLs();
  void buildGraph();
  void fillGraph(Function &function);
  void transitiveClosure();
  void buildBackwardGraph();

private:
  typedef std::map<BasicBlock *, BlockVector> GraphMap;
  PostDominatorTree *pdt;
  GraphMap forwardGraph;
  GraphMap backwardGraph;
  std::vector<std::pair<BasicBlock *, BasicBlock *> > s;
  BlockVector ls;
};

#endif
