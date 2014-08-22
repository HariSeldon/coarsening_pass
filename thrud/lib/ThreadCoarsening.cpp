#include "thrud/ThreadCoarsening.h"

#include "thrud/DivergenceAnalysis.h"

#include "thrud/DataTypes.h"
#include "thrud/NDRange.h"
#include "thrud/Utils.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueMap.h"

#include "llvm/Pass.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Twine.h"

#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/RegionInfo.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Transforms/Scalar.h"

#include <utility>

using namespace llvm;

// Command line options.
extern cl::opt<unsigned int> CoarseningDirectionCL;
cl::opt<unsigned int> CoarseningFactorCL("coarsening-factor", cl::init(1),
                                         cl::Hidden,
                                         cl::desc("The coarsening factor"));
cl::opt<unsigned int> CoarseningStrideCL("coarsening-stride", cl::init(1),
                                         cl::Hidden,
                                         cl::desc("The coarsening stride"));
cl::opt<std::string> KernelNameCL("kernel-name", cl::init(""), cl::Hidden,
                                  cl::desc("Name of the kernel to coarsen"));
cl::opt<ThreadCoarsening::DivRegionOption> DivRegionOptionCL(
    "div-region-mgt", cl::init(ThreadCoarsening::FullReplication), cl::Hidden,
    cl::desc("Divergent region management"),
    cl::values(clEnumValN(ThreadCoarsening::FullReplication, "classic",
                          "Replicate full region"),
               clEnumValN(ThreadCoarsening::TrueBranchMerging, "merge-true",
                          "Merge true branch"),
               clEnumValN(ThreadCoarsening::FalseBranchMerging, "merge-false",
                          "Merge false branch"),
               clEnumValN(ThreadCoarsening::FullMerging, "merge",
                          "Merge both true and false branches"),
               clEnumValEnd));

//------------------------------------------------------------------------------
ThreadCoarsening::ThreadCoarsening() : FunctionPass(ID) {}

//------------------------------------------------------------------------------
void ThreadCoarsening::getAnalysisUsage(AnalysisUsage &au) const {
  au.addRequired<LoopInfo>();
  au.addRequired<SingleDimDivAnalysis>();
  au.addRequired<PostDominatorTree>();
  au.addRequired<DominatorTreeWrapperPass>();
  au.addRequired<NDRange>();
}

//------------------------------------------------------------------------------
bool ThreadCoarsening::runOnFunction(Function &F) {
  // Apply the pass to kernels only.
  if (!isKernel((const Function *)&F))
    return false;

  // Apply the pass to the selected kernel only.
  std::string FunctionName = F.getName();
  if (KernelNameCL != "" && FunctionName != KernelNameCL)
    return false;

  // Get command line options.
  direction = CoarseningDirectionCL;
  factor = CoarseningFactorCL;
  stride = CoarseningStrideCL;
  divRegionOption = DivRegionOptionCL;

  // Perform analysis.
  loopInfo = &getAnalysis<LoopInfo>();
  pdt = &getAnalysis<PostDominatorTree>();
  dt = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  sdda = &getAnalysis<SingleDimDivAnalysis>();
  ndr = &getAnalysis<NDRange>();

  // Transform the kernel.
  init();
  scaleNDRange();
  coarsenFunction();
  replacePlaceholders();

  return true;
}

void ThreadCoarsening::init() {
  cMap.clear();
  phMap.clear();
  phReplacementMap.clear();
}

//------------------------------------------------------------------------------
char ThreadCoarsening::ID = 0;
static RegisterPass<ThreadCoarsening>
    X("tc", "OpenCL Thread Coarsening Transformation Pass");
