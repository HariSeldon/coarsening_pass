#ifndef SYMBOLIC_EXECUTION_H
#define SYMBOLIC_EXECUTION_H

#include "thrud/NDRangeSpace.h"

#include "llvm/Pass.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/ScalarEvolution.h"

#include "llvm/IR/InstVisitor.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace llvm {
class Function;
class StoreInst;
class LoadInst;
}

class MultiDimDivAnalysis;
class NDRange;
class OCLEnv;
class SubscriptAnalysis;

/// Collect information about the kernel function.
namespace {
class SymbolicExecution : public FunctionPass,
                          public InstVisitor<SymbolicExecution> {

  friend class InstVisitor<SymbolicExecution>;

public:
  static char ID;
  SymbolicExecution();
  ~SymbolicExecution();

  virtual bool runOnFunction(Function &F);
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;

public:
  std::vector<int> loadTransactions;
  std::vector<int> storeTransactions;

  std::vector<int> loopLoadTransactions;
  std::vector<int> loopStoreTransactions;

  std::vector<int> loadBankConflicts;
  std::vector<int> storeBankConflicts;

  std::vector<int> loopLoadBankConflicts;
  std::vector<int> loopStoreBankConflicts;

private:
  void memoryAccessAnalysis(BasicBlock &block, std::vector<int> &loadTrans,
                            std::vector<int> &storeTrans);
  void init();
  void initBuffers();
  void initOCLSpace();
  void visitLoadInst(LoadInst &loadInst);
  void visitStoreInst(StoreInst &storeInst);
  void visitMemoryInst(Value *pointer, std::vector<int> &resultVector);
  void visitLocalMemoryInst(Value *pointer, std::vector<int> &resultVector);
  void dump();

private:
  ScalarEvolution *scalarEvolution;
  SubscriptAnalysis *subscriptAnalysis;
  OCLEnv *ocl;
  NDRange *ndr;
  LoopInfo *loopInfo;
  NDRangeSpace ndrSpace;
};
}

#endif
