#include "thrud/FeatureCollector.h"

#include "thrud/DataTypes.h"
#include "thrud/Graph.h"
#include "thrud/MathUtils.h"
#include "thrud/OCLEnv.h"
#include "thrud/Utils.h"
#include "thrud/SubscriptAnalysis.h"

#include "thrud/ILPComputation.h"
#include "thrud/MLPComputation.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Function.h"

#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/PostDominators.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/YAMLTraits.h"

#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>

using namespace llvm;

using yaml::MappingTraits;
using yaml::SequenceTraits;
using yaml::IO;
using yaml::Output;

namespace llvm {
namespace yaml {

//------------------------------------------------------------------------------
template <> struct MappingTraits<FeatureCollector> {
  static void mapping(IO &io, FeatureCollector &collector) {
    for (auto &iter : collector.instTypes) {
      io.mapRequired(iter.first.c_str(), iter.second);
    }
    // Instructions per block.
    io.mapRequired("instsPerBlock", collector.blockInsts);

    // Dump Phi nodes.
    std::vector<int> args;

    for (auto &iter : collector.blockPhis) {
      std::vector<std::string> phis = iter.second;
      int argSum = 0;
      for (auto &phiIter : phis) {
        int argNumber = collector.phiArgs[phiIter];
        argSum += argNumber;
      }
      args.push_back(argSum);
    }

    io.mapRequired("phiArgs", args);
    io.mapRequired("ilpPerBlock", collector.blockILP);
    io.mapRequired("mlpPerBlock", collector.blockMLP);
    io.mapRequired("avgLiveRange", collector.avgLiveRange);
    io.mapRequired("aliveOut", collector.aliveOutBlocks);
  }
};

//------------------------------------------------------------------------------
template <> struct MappingTraits<std::pair<float, float> > {
  static void mapping(IO &io, std::pair<float, float> &avgVar) {
    io.mapRequired("avg", avgVar.first);
    io.mapRequired("var", avgVar.second);
  }
};

//------------------------------------------------------------------------------
// Sequence of ints.
template <> struct SequenceTraits<std::vector<int> > {
  static size_t size(IO &, std::vector<int> &seq) { return seq.size(); }
  static int &element(IO &, std::vector<int> &seq, size_t index) {
    if (index >= seq.size())
      seq.resize(index + 1);
    return seq[index];
  }

  static const bool flow = true;
};

//------------------------------------------------------------------------------
// Sequence of floats.
template <> struct SequenceTraits<std::vector<float> > {
  static size_t size(IO &, std::vector<float> &seq) { return seq.size(); }
  static float &element(IO &, std::vector<float> &seq, size_t index) {
    if (index >= seq.size())
      seq.resize(index + 1);
    return seq[index];
  }

  static const bool flow = true;
};

//------------------------------------------------------------------------------
// Sequence of pairs.
template <> struct SequenceTraits<std::vector<std::pair<float, float> > > {
  static size_t size(IO &, std::vector<std::pair<float, float> > &seq) {
    return seq.size();
  }
  static std::pair<float, float> &
  element(IO &, std::vector<std::pair<float, float> > &seq, size_t index) {
    if (index >= seq.size())
      seq.resize(index + 1);
    return seq[index];
  }

  static const bool flow = true;
};
}
}

//------------------------------------------------------------------------------
FeatureCollector::FeatureCollector() {
// Instruction-specific counters.
#define HANDLE_INST(N, OPCODE, CLASS) instTypes[#OPCODE] = 0;
#include "llvm/IR/Instruction.def"

  // Initialize all counters.
  instTypes["insts"] = 0;
  instTypes["blocks"] = 0;
  instTypes["edges"] = 0;
  instTypes["criticalEdges"] = 0;
  instTypes["condBranches"] = 0;
  instTypes["uncondBranches"] = 0;
  instTypes["fourB"] = 0;
  instTypes["eightB"] = 0;
  instTypes["fps"] = 0;
  instTypes["vector2"] = 0;
  instTypes["vector4"] = 0;
  instTypes["vectorOperands"] = 0;
  instTypes["localLoads"] = 0;
  instTypes["localStores"] = 0;
  instTypes["mathFunctions"] = 0;
  instTypes["barriers"] = 0;
  instTypes["args"] = 0;
  instTypes["divRegions"] = 0;
  instTypes["divInsts"] = 0;
  instTypes["divRegionInsts"] = 0;
  instTypes["uniformLoads"] = 0;
}

//------------------------------------------------------------------------------
void FeatureCollector::computeILP(BasicBlock *block) {
  blockILP.push_back(getILP(block));
}

//------------------------------------------------------------------------------
void FeatureCollector::computeMLP(BasicBlock *block) {
  blockMLP.push_back(getMLP(block));
}

//------------------------------------------------------------------------------
void FeatureCollector::countIncomingEdges(const BasicBlock &block) {
  const BasicBlock *tmpBlock = (const BasicBlock *)&block;
  const_pred_iterator first = pred_begin(tmpBlock), last = pred_end(tmpBlock);
  blockIncoming[block.getName()] = std::distance(first, last);
}

//------------------------------------------------------------------------------
void FeatureCollector::countOutgoingEdges(const BasicBlock &block) {
  blockOutgoing[block.getName()] = block.getTerminator()->getNumSuccessors();
}

//------------------------------------------------------------------------------
void FeatureCollector::countInstsBlock(const BasicBlock &block) {
  blockInsts.push_back(static_cast<int>(block.getInstList().size()));
}

//------------------------------------------------------------------------------
void FeatureCollector::countEdges(const Function &function) {
  int criticalEdges = 0;

  int edges = std::accumulate(function.begin(), function.end(), 0,
                              [](int result, const BasicBlock &block) {
    return result + block.getTerminator()->getNumSuccessors();
  });

  for (auto &block : function) {
    const TerminatorInst *termInst = block.getTerminator();
    int termNumber = termInst->getNumSuccessors();

    for (int index = 0; index < termNumber; ++index) {
      criticalEdges += isCriticalEdge(termInst, index);
    }
  }

  instTypes["edges"] = edges;
  instTypes["criticalEdges"] = criticalEdges;
}

//------------------------------------------------------------------------------
void FeatureCollector::countBranches(const Function &function) {
  int condBranches = 0;
  int uncondBranches = 0;
  for (auto &block : function) {
    const TerminatorInst *term = block.getTerminator();
    if (const BranchInst *branch = dyn_cast<BranchInst>(term)) {
      if (branch->isConditional())
        ++condBranches;
      else
        ++uncondBranches;
    }
  }

  instTypes["condBranches"] = condBranches;
  instTypes["uncondBranches"] = uncondBranches;
}

//------------------------------------------------------------------------------
void FeatureCollector::countPhis(const BasicBlock &block) {
  std::vector<std::string> names;
  for (auto inst = block.begin(); isa<PHINode>(inst); ++inst) {
    std::string name = inst->getName();
    int argCount = inst->getNumOperands();

    phiArgs[name] = argCount;
    names.push_back(name);
  }

  blockPhis[block.getName()] = names;
}

//------------------------------------------------------------------------------
void FeatureCollector::countConstants(const BasicBlock &block) {
  int fourB = instTypes["fourB"];
  int eightB = instTypes["eightB"];
  int fps = instTypes["fps"];

  for (const auto &inst : block) {
    for (Instruction::const_op_iterator opIter = inst.op_begin(),
                                        opEnd = inst.op_end();
         opIter != opEnd; ++opIter) {
      const Value *operand = opIter->get();
      if (const ConstantInt *constInt = dyn_cast<ConstantInt>(operand)) {
        if (constInt->getBitWidth() == 32)
          ++fourB;

        if (constInt->getBitWidth() == 64)
          ++eightB;
      }

      if (isa<ConstantFP>(operand))
        ++fps;
    }
  }

  instTypes["fourB"] = fourB;
  instTypes["eightB"] = eightB;
  instTypes["fps"] = fps;
}

//------------------------------------------------------------------------------
void FeatureCollector::countBarriers(const BasicBlock &block) {
  for (const auto &inst : block) {
    if (const CallInst *callInst = dyn_cast<CallInst>(&inst)) {
      const Function *function = callInst->getCalledFunction();
      if (function == nullptr)
        continue;
      if (function->getName() == "barrier") {
        safeIncrement(instTypes, "barriers");
      }
    }
  }
}

//------------------------------------------------------------------------------
void FeatureCollector::countMathFunctions(const BasicBlock &block) {
  for (const auto &inst : block) {
    if (const CallInst *callInst = dyn_cast<CallInst>(&inst)) {
      const Function *function = callInst->getCalledFunction();
      if (function == nullptr)
        continue;
      if (isMathName(function->getName())) {
        safeIncrement(instTypes, "mathFunctions");
      }
    }
  }
}

//------------------------------------------------------------------------------
void FeatureCollector::countLocalMemoryUsage(const BasicBlock &block) {
  for (const auto &inst : block) {
    if (const LoadInst *loadInst = dyn_cast<LoadInst>(&inst)) {
      if (loadInst->getPointerAddressSpace() == OCLEnv::LOCAL_AS)
        safeIncrement(instTypes, "localLoads");
    }
    if (const StoreInst *storeInst = dyn_cast<StoreInst>(&inst)) {
      if (storeInst->getPointerAddressSpace() == OCLEnv::LOCAL_AS)
        safeIncrement(instTypes, "localStores");
    }
  }
}

//------------------------------------------------------------------------------
void FeatureCollector::countDivInsts(Function &function,
                                     MultiDimDivAnalysis *mdda,
                                     SingleDimDivAnalysis *sdda) {
  // Insts in divergent regions.
  RegionVector &regions = mdda->getDivRegions();
  int divRegionInsts = std::accumulate(regions.begin(), regions.end(), 0,
                                       [](int result, DivergentRegion *region) {
    return result + region->size();
  });

  // Count uniform loads.
  int uniformLoads = 0;
  for (inst_iterator iter = inst_begin(function), iterEnd = inst_end(function);
       iter != iterEnd; ++iter) {
    Instruction *inst = &*iter;
    uniformLoads += isa<LoadInst>(inst) * !sdda->isDivergent(inst);
  }

  instTypes["divRegionInsts"] = divRegionInsts;
  instTypes["uniformLoads"] = uniformLoads;
  instTypes["divRegions"] = mdda->getDivRegions().size();
  instTypes["divInsts"] = mdda->getDivInsts().size();
}

//------------------------------------------------------------------------------
void FeatureCollector::countArgs(const Function &function) {
  instTypes["args"] = function.arg_size();
}

//------------------------------------------------------------------------------
int computeLiveRange(Instruction *inst) {
  Instruction *lastUser = findLastUser(inst);

  if (lastUser == nullptr)
    return inst->getParent()->size();

  assert(lastUser->getParent() == inst->getParent() &&
         "Different basic blocks");

  BasicBlock::iterator begin(inst), end(lastUser);

  return std::distance(begin, end);
}

//------------------------------------------------------------------------------
void FeatureCollector::livenessAnalysis(BasicBlock &block) {
  int aliveValues = 0;
  std::vector<int> ranges;

  for (auto &inst : block) {
    if (!inst.hasName())
      continue;

    bool isUsedElseWhere = isUsedOutsideOfDefiningBlock(&inst);
    aliveValues += isUsedElseWhere;

    if (!isUsedElseWhere) {
      int liveRange = computeLiveRange(&inst);
      ranges.push_back(liveRange);
    }
  }

  avgLiveRange.push_back(getAverage(ranges));
  aliveOutBlocks.push_back(aliveValues);
}

////------------------------------------------------------------------------------
//void FeatureCollector::countDimensions(NDRange *NDR) {
//  InstVector dir0 = NDR->getTids(0);
//  InstVector dir1 = NDR->getTids(1);
//  InstVector dir2 = NDR->getTids(2);
//
//  int dimensionNumber =
//      (dir0.size() != 0) + (dir1.size() != 0) + (dir2.size() != 0);
//
//  instTypes["dimensions"] = dimensionNumber;
//}

//------------------------------------------------------------------------------
void FeatureCollector::dump() {
  Output yout(llvm::outs());
  yout << *this;
}

//------------------------------------------------------------------------------
void FeatureCollector::loopCountEdges(const Function &function, LoopInfo *LI) {
  int edges = std::accumulate(function.begin(), function.end(), 0,
                              [LI](int result, const BasicBlock &block) {
    return result +
           (isInLoop(&block, LI) * block.getTerminator()->getNumSuccessors());
  });

  int criticalEdges = 0;
  for (auto &block : function) {
    if (!isInLoop(&block, LI))
      continue;

    const TerminatorInst *termInst = block.getTerminator();
    int termNumber = termInst->getNumSuccessors();

    for (int index = 0; index < termNumber; ++index) {
      criticalEdges += isCriticalEdge(termInst, index);
    }
  }

  instTypes["edges"] = edges;
  instTypes["criticalEdges"] = criticalEdges;
}

//------------------------------------------------------------------------------
void FeatureCollector::loopCountBranches(const Function &function,
                                         LoopInfo *LI) {
  int condBranches = 0;
  int uncondBranches = 0;
  for (auto &block : function) {

    if (!isInLoop(&block, LI))
      continue;

    const TerminatorInst *term = block.getTerminator();
    if (const BranchInst *branch = dyn_cast<BranchInst>(term)) {
      if (branch->isConditional() == true)
        ++condBranches;
      else
        ++uncondBranches;
    }
  }

  instTypes["condBranches"] = condBranches;
  instTypes["uncondBranches"] = uncondBranches;
}
//------------------------------------------------------------------------------
void FeatureCollector::loopCountDivInsts(Function &function,
                                         MultiDimDivAnalysis *mdda,
                                         SingleDimDivAnalysis *sdda,
                                         LoopInfo *LI) {
  // Count divergent regions.
  RegionVector &regions = mdda->getDivRegions();
  InstVector divInsts = mdda->getDivInsts();
  
  // Count number of divergent instructions.
  int divInstsCounter = std::accumulate(divInsts.begin(), divInsts.end(), 0,
                                    [LI](int result, Instruction *inst) {
    return result + (isInLoop(inst, LI));
  });

  // Count number of divergent regions.
  int divRegionsCounter = std::count_if(regions.begin(), regions.end(),
                                        [LI](DivergentRegion *region) {
    return isInLoop(region->getHeader(), LI);
  });

  // Count number of instructions in divergent regions.
  int divRegionInsts =
      std::accumulate(regions.begin(), regions.end(), 0,
                      [LI](int result, DivergentRegion *region) {
        return (result + (region->size() * isInLoop(region->getHeader(), LI)));
      });

  // Count uniform loads.
  int uniformLoads = 0;
  for (inst_iterator iter = inst_begin(function), iterEnd = inst_end(function);
       iter != iterEnd; ++iter) {
    Instruction *inst = &*iter;
    uniformLoads += isInLoop(inst, LI) * isa<LoadInst>(inst) * !sdda->isDivergent(inst);
  }

  instTypes["divInsts"] = divInstsCounter;
  instTypes["divRegions"] = divRegionsCounter;
  instTypes["divRegionInsts"] = divRegionInsts;
  instTypes["uniformLoads"] = uniformLoads;
}
