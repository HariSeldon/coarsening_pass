#include "thrud/DivergenceAnalysis.h"

#include "thrud/DivergentRegion.h"
#include "thrud/Utils.h"

#include "llvm/Pass.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Function.h"

#include "llvm/ADT/Statistic.h"

#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <utility>

using namespace llvm;

extern cl::opt<unsigned int> CoarseningDirectionCL;

// Support functions.
// -----------------------------------------------------------------------------
void findUsesOf(Instruction *inst, InstSet &result);
bool isOutermost(Instruction *inst, RegionVector &regions);
bool isOutermost(DivergentRegion *region, RegionVector &regions);

// DivergenceAnalysis.
// -----------------------------------------------------------------------------
void DivergenceAnalysis::init() {
  divInsts.clear();
  outermostDivInsts.clear();
  divBranches.clear();
  regions.clear();
  outermostRegions.clear();
}

//------------------------------------------------------------------------------
InstVector DivergenceAnalysis::getTids() {
  // This must be overriden by all subclasses.
  return InstVector();
}

//------------------------------------------------------------------------------
void DivergenceAnalysis::performAnalysis() {
  InstVector seeds = getTids();
  InstSet worklist(seeds.begin(), seeds.end());

  while (!worklist.empty()) {
    auto iter = worklist.begin();
    Instruction *inst = *iter;
    worklist.erase(iter);
    divInsts.push_back(inst);

    InstSet users;

    // Manage branches.
    if (isa<BranchInst>(inst)) {
      BasicBlock *block = findImmediatePostDom(inst->getParent(), pdt);
      for (auto inst = block->begin(); isa<PHINode>(inst); ++inst) {
        users.insert(inst);
      }
    }

    findUsesOf(inst, users);
    // Add users of the current instruction to the work list.
    for (InstSet::iterator iter = users.begin(), iterEnd = users.end();
         iter != iterEnd; ++iter) {
      if (!isPresent(*iter, divInsts))
        worklist.insert(*iter);
    }
  }
}

//------------------------------------------------------------------------------
void DivergenceAnalysis::findBranches() {
  // Find all branches.
  std::copy_if(divInsts.begin(), divInsts.end(),
               std::back_inserter(divBranches),
               [](Instruction *inst) { return isa<BranchInst>(inst); });
}

//------------------------------------------------------------------------------
RegionVector cleanUpRegions(RegionVector &regions, const DominatorTree *dt) {
  RegionVector result;

  for (size_t index1 = 0; index1 < regions.size(); ++index1) {
    DivergentRegion *region1 = regions[index1];
    BlockVector &blocks1 = region1->getBlocks();
  
    bool toAdd = true;

    for (size_t index2 = 0; index2 < regions.size(); ++index2) {
      if(index2 == index1) 
        break;

      DivergentRegion *region2 = regions[index2];
      BlockVector &blocks2 = region2->getBlocks();

      if (is_permutation(blocks1.begin(), blocks1.end(), blocks2.begin()) &&
        dt->dominates(region2->getHeader(), region1->getHeader())) {
          toAdd = false;
          break; 
      }
    }

    if(toAdd) 
      result.push_back(region1); 
  }

  return result;
}

//------------------------------------------------------------------------------
void DivergenceAnalysis::findRegions() {
  for (auto branch : divBranches) {
    BasicBlock *header = branch->getParent();
    BasicBlock *exiting = findImmediatePostDom(header, pdt);

    if (loopInfo->isLoopHeader(header)) {
      Loop *loop = loopInfo->getLoopFor(header);
      if (loop == loopInfo->getLoopFor(exiting))
        exiting = loop->getExitBlock();
    }

    regions.push_back(new DivergentRegion(header, exiting));
  }

  // Remove redundant regions. The ones coming from loops.
  regions = cleanUpRegions(regions, dt);
}

//------------------------------------------------------------------------------
// This is called only when the outermost instructions are acutally requested,
// ie. during coarsening. This is done to be sure that this instructions are
// computed after the extraction of divergent regions from the CFG.
void DivergenceAnalysis::findOutermostInsts(InstVector &insts,
                                            RegionVector &regions,
                                            InstVector &result) {
  result.clear();
  for (auto inst : insts) {
    if (isOutermost(inst, regions)) {
      result.push_back(inst);
    }
  }

  // Remove from result all the calls to builtin functions.
  InstVector oclIds = ndr->getTids();
  InstVector tmp;

  size_t oldSize = result.size();

  std::sort(result.begin(), result.end());
  std::sort(oclIds.begin(), oclIds.end());
  std::set_difference(result.begin(), result.end(), oclIds.begin(),
                      oclIds.end(), std::back_inserter(tmp));
  result.swap(tmp);

  assert(result.size() <= oldSize && "Wrong set difference");
}

//------------------------------------------------------------------------------
void DivergenceAnalysis::findOutermostRegions() {
  outermostRegions.clear();
  for (auto region : regions) {
    if (isOutermost(region, regions)) {
      outermostRegions.push_back(region);
    }
  }
}

// Public functions.
//------------------------------------------------------------------------------
InstVector &DivergenceAnalysis::getDivInsts() { return divInsts; }

//------------------------------------------------------------------------------
InstVector &DivergenceAnalysis::getOutermostDivInsts() {
  // Use memoization.
  if (outermostDivInsts.empty())
    findOutermostInsts(divInsts, regions, outermostDivInsts);
  return outermostDivInsts;
}

//------------------------------------------------------------------------------
InstVector DivergenceAnalysis::getDivInsts(DivergentRegion *region) {
  InstVector tmp;
  DivergentRegion &r = *region;

  for (InstVector::iterator iter = divInsts.begin(), iterEnd = divInsts.end();
       iter != iterEnd; ++iter) {
    Instruction *inst = *iter;
    if (containsInternally(r, inst)) {
      tmp.push_back(inst);
    }
  }

  RegionVector internalRegions = getDivRegions(region);
  InstVector result;
  findOutermostInsts(tmp, internalRegions, result);
  return result;
}

//------------------------------------------------------------------------------
RegionVector &DivergenceAnalysis::getDivRegions() { return regions; }

//------------------------------------------------------------------------------
RegionVector &DivergenceAnalysis::getOutermostDivRegions() {
  // Use memoization.
  if (outermostRegions.empty()) {
    findOutermostRegions();
  }
  return outermostRegions;
}

//------------------------------------------------------------------------------
RegionVector DivergenceAnalysis::getDivRegions(DivergentRegion *region) {

  RegionVector tmpVector;
  DivergentRegion &r = *region;

  for (auto currentRegion : regions) {
    if (containsInternally(r, currentRegion)) {
      tmpVector.push_back(currentRegion);
    }
  }

  RegionVector result;
  for (auto currentRegion : tmpVector) {
    if (isOutermost(currentRegion, tmpVector)) {
      result.push_back(currentRegion);
    }
  }

  return result;
}

//------------------------------------------------------------------------------
bool DivergenceAnalysis::isDivergent(Instruction *inst) {
  return isPresent(inst, divInsts);
}

// Support functions.
//------------------------------------------------------------------------------
void findUsesOf(Instruction *inst, InstSet &result) {
  for (auto userIter = inst->user_begin(), userEnd = inst->user_end();
       userIter != userEnd; ++userIter) {
    if (Instruction *userInst = dyn_cast<Instruction>(*userIter)) {
      result.insert(userInst);
    }
  }
}

//------------------------------------------------------------------------------
bool isOutermost(Instruction *inst, RegionVector &regions) {
  bool result = false;
  for (RegionVector::const_iterator iter = regions.begin(),
                                    iterEnd = regions.end();
       iter != iterEnd; ++iter) {
    DivergentRegion *region = *iter;
    result |= contains(*region, inst);
  }
  return !result;
}

//------------------------------------------------------------------------------
bool isOutermost(DivergentRegion *region, RegionVector &regions) {
  Instruction *inst = region->getHeader()->getTerminator();
  bool result = false;
  for (RegionVector::const_iterator iter = regions.begin(),
                                    iterEnd = regions.end();
       iter != iterEnd; ++iter) {
    DivergentRegion *region = *iter;
    result |= containsInternally(*region, inst);
  }
  return !result;
}

// SingleDimDivAnalysis
//------------------------------------------------------------------------------
SingleDimDivAnalysis::SingleDimDivAnalysis() : FunctionPass(ID) {}

void SingleDimDivAnalysis::getAnalysisUsage(AnalysisUsage &au) const {
  au.addRequired<LoopInfo>();
  au.addPreserved<LoopInfo>();
  au.addRequired<PostDominatorTree>();
  au.addRequired<DominatorTreeWrapperPass>();
  au.addRequired<NDRange>();
  au.addRequired<ControlDependenceAnalysis>();
  au.setPreservesAll();
}

bool SingleDimDivAnalysis::runOnFunction(Function &functionRef) {
  Function *function = (Function *)&functionRef;
  // Apply the pass to kernels only.
  if (!isKernel(function))
    return false;

  init();
  pdt = &getAnalysis<PostDominatorTree>();
  dt = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  loopInfo = &getAnalysis<LoopInfo>();
  ndr = &getAnalysis<NDRange>();
  cda = &getAnalysis<ControlDependenceAnalysis>();

  performAnalysis();
  findBranches();
  findRegions();

  return false;
}

InstVector SingleDimDivAnalysis::getTids() {
  return ndr->getTids(CoarseningDirectionCL);
}

char SingleDimDivAnalysis::ID = 0;
static RegisterPass<SingleDimDivAnalysis> X("sdda",
                                            "Single divergence analysis");

// MultiDimDivAnalysis
//------------------------------------------------------------------------------
MultiDimDivAnalysis::MultiDimDivAnalysis() : FunctionPass(ID) {}

void MultiDimDivAnalysis::getAnalysisUsage(AnalysisUsage &au) const {
  au.addRequired<LoopInfo>();
  au.addRequired<PostDominatorTree>();
  au.addRequired<DominatorTreeWrapperPass>();
  au.addRequired<NDRange>();
  au.addRequired<ControlDependenceAnalysis>();
  au.setPreservesAll();
}

bool MultiDimDivAnalysis::runOnFunction(Function &functionRef) {
  Function *function = (Function *)&functionRef;
  // Apply the pass to kernels only.
  if (!isKernel(function))
    return false;

  init();
  pdt = &getAnalysis<PostDominatorTree>();
  dt = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  loopInfo = &getAnalysis<LoopInfo>();
  ndr = &getAnalysis<NDRange>();
  cda = &getAnalysis<ControlDependenceAnalysis>();

  performAnalysis();
  findBranches();
  findRegions();

  return false;
}

InstVector MultiDimDivAnalysis::getTids() { return ndr->getTids(); }

char MultiDimDivAnalysis::ID = 0;
static RegisterPass<MultiDimDivAnalysis>
    Y("mdda", "Multidimensional divergence analysis");
