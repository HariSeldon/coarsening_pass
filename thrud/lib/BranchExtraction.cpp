#include "thrud/BranchExtraction.h"

#include "thrud/DataTypes.h"
#include "thrud/DivergenceAnalysis.h"
#include "thrud/DivergentRegion.h"
#include "thrud/Utils.h"

#include "llvm/Pass.h"

#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/PostDominators.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <algorithm>
#include <functional>
#include <utility>

using namespace llvm;

extern cl::opt<std::string> KernelNameCL;
cl::opt<int> CoarseningDirectionCL("coarsening-direction", cl::init(0),
                                   cl::Hidden,
                                   cl::desc("The coarsening direction"));

//------------------------------------------------------------------------------
BranchExtraction::BranchExtraction() : FunctionPass(ID) {}

//------------------------------------------------------------------------------
void BranchExtraction::getAnalysisUsage(AnalysisUsage &au) const {
  au.addRequired<LoopInfo>();
  au.addRequired<SingleDimDivAnalysis>();
  au.addRequired<PostDominatorTree>();
  au.addRequired<DominatorTreeWrapperPass>();
  au.addPreserved<SingleDimDivAnalysis>();
}

//------------------------------------------------------------------------------
bool BranchExtraction::runOnFunction(Function &F) {
  // Apply the pass to kernels only.
  if (!isKernel((const Function *)&F))
    return false;

  std::string FunctionName = F.getName();
  if (KernelNameCL != "" && FunctionName != KernelNameCL)
    return false;

  // Perform analyses.
  loopInfo = &getAnalysis<LoopInfo>();
  dt = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  pdt = &getAnalysis<PostDominatorTree>();

  sdda = &getAnalysis<SingleDimDivAnalysis>();
  RegionVector &regions = sdda->getDivRegions();

  // This is terribly inefficient.
  for (auto region : regions) {
    BasicBlock *newExiting = findImmediatePostDom(region->getHeader(), pdt);
    region->setExiting(newExiting);
    region->fillRegion();
    extractBranches(region);
    region->fillRegion();
    isolateRegion(region);
    region->fillRegion();
    region->findAliveValues();
    dt->recalculate(F);
    pdt->DT->recalculate(F);
  }

  std::for_each(
      regions.begin(), regions.end(),
      [this](DivergentRegion *region) { region->fillRegion(); });

  return regions.size() != 0;
}

//------------------------------------------------------------------------------
// Isolate the exiting block from the rest of the graph.
// If it has incoming edges coming from outside the current region
// create a new exiting block for the region.
void BranchExtraction::extractBranches(DivergentRegion *region) {
  BasicBlock *header = region->getHeader();
  BasicBlock *exiting = region->getExiting();
  BasicBlock *newHeader = nullptr;

  if (!loopInfo->isLoopHeader(header))
    newHeader = SplitBlock(header, header->getTerminator(), this);
  else {
    newHeader = header;
    Loop *loop = loopInfo->getLoopFor(header);
    if (loop == loopInfo->getLoopFor(exiting)) {
      exiting = loop->getExitBlock();
      region->setExiting(exiting);
    }
  }

  Instruction *firstNonPHI = exiting->getFirstNonPHI();
  BasicBlock *newExiting = SplitBlock(exiting, firstNonPHI, this);
  region->setHeader(newHeader);

  // Check is a region in the has as header exiting.
  // If so replace it with new exiting.
  RegionVector &regions = sdda->getDivRegions();

  std::for_each(regions.begin(), regions.end(),
                [exiting, newExiting](DivergentRegion *region) {
    if (region->getHeader() == exiting)
      region->setHeader(newExiting);
  });
}

// -----------------------------------------------------------------------------
void BranchExtraction::isolateRegion(DivergentRegion *region) {
  BasicBlock *exiting = region->getExiting();

  // If the header dominates the exiting bail out.
  if (dt->dominates(region->getHeader(), region->getExiting()))
    return;

  // TODO.
  // Verify that the incoming branch from outside is pointing to the exiting
  // block.

  // Create a new exiting block.
  BasicBlock *newExiting = BasicBlock::Create(
      exiting->getContext(), exiting->getName() + Twine(".extracted"),
      exiting->getParent(), exiting);
  BranchInst::Create(exiting, newExiting);

  // All the blocks in the region pointing to the exiting are redirected to the
  // new exiting.
  for (auto iter = region->begin(), iterEnd = region->end(); iter != iterEnd;
       ++iter) {
    TerminatorInst *terminator = (*iter)->getTerminator();
    for (unsigned int index = 0; index < terminator->getNumSuccessors();
         ++index) {
      if (terminator->getSuccessor(index) == exiting) {
        terminator->setSuccessor(index, newExiting);
      }
    }
  }

  // 'newExiting' will contain the phi working on the values from the blocks
  // in the region.
  // 'Exiting' will contain the phi working on the values from the blocks
  // outside and in the region.
  PhiVector oldPhis = getPHIs(exiting);

  PhiVector newPhis;
  PhiVector exitPhis;

  InstVector &divInsts = sdda->getDivInsts();

  for (auto phi: oldPhis) {
    PHINode *newPhi = PHINode::Create(phi->getType(), 0,
                                      phi->getName() + Twine(".new_exiting"),
                                      newExiting->begin());
    PHINode *exitPhi = PHINode::Create(phi->getType(), 0,
                                       phi->getName() + Twine(".old_exiting"),
                                       exiting->begin());
    for (unsigned int index = 0; index < phi->getNumIncomingValues(); ++index) {
      BasicBlock *BB = phi->getIncomingBlock(index);
      if (contains(*region, BB))
        newPhi->addIncoming(phi->getIncomingValue(index), BB);
      else
        exitPhi->addIncoming(phi->getIncomingValue(index), BB);
    }
    newPhis.push_back(newPhi);
    exitPhis.push_back(exitPhi);

    // Update divInsts.
    if (std::find(divInsts.begin(), divInsts.end(),
                  static_cast<Instruction *>(phi)) !=
        divInsts.end()) {
      divInsts.push_back(newPhi);
      divInsts.push_back(exitPhi);
    }
  }

  unsigned int phiNumber = newPhis.size();
  for (unsigned int phiIndex = 0; phiIndex < phiNumber; ++phiIndex) {
    // Add the edge coming from the 'newExiting' block to the phi nodes in
    // Exiting.
    PHINode *exitPhi = exitPhis[phiIndex];
    PHINode *newPhi = newPhis[phiIndex];
    exitPhi->addIncoming(newPhi, newExiting);

    // Update all the references to the old Phis to the new ones.
    oldPhis[phiIndex]->replaceAllUsesWith(exitPhi);
  }

  // Delete the old phi nodes.
  for (auto toDelete: oldPhis) {
    // Update divInsts.
    InstVector::iterator iter = std::find(divInsts.begin(), divInsts.end(),
                                          static_cast<Instruction *>(toDelete));
    if (iter != divInsts.end()) {
      divInsts.erase(iter);
    }

    toDelete->eraseFromParent();
  }

  region->setExiting(newExiting);
}

//------------------------------------------------------------------------------
char BranchExtraction::ID = 0;
static RegisterPass<BranchExtraction> X("be", "Extract divergent regions");
