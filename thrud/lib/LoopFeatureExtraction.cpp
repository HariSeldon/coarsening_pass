#include "thrud/LoopFeatureExtraction.h"

#include "thrud/DivergenceAnalysis.h"
#include "thrud/NDRange.h"
#include "thrud/OCLEnv.h"
#include "thrud/Utils.h"

cl::opt<std::string> loopKernelName("count-loop-kernel-name", cl::init(""),
                                    cl::Hidden,
                                    cl::desc("Name of the kernel to analyze"));

extern cl::opt<unsigned int> CoarseningDirectionCL;

char OpenCLLoopFeatureExtractor::ID = 0;
static RegisterPass<OpenCLLoopFeatureExtractor>
    X("opencl-loop-instcount", "Collect opencl features in loops");

//------------------------------------------------------------------------------
bool OpenCLLoopFeatureExtractor::runOnFunction(Function &function) {
  if (function.getName() != loopKernelName)
    return false;

  PDT = &getAnalysis<PostDominatorTree>();
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  MDDA = &getAnalysis<MultiDimDivAnalysis>();
  SDDA = &getAnalysis<SingleDimDivAnalysis>();
  LI = &getAnalysis<LoopInfo>();
  SE = &getAnalysis<ScalarEvolution>();

  visit(function);
  collector.dump();
  return false;
}

//------------------------------------------------------------------------------
void OpenCLLoopFeatureExtractor::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MultiDimDivAnalysis>();
  AU.addRequired<SingleDimDivAnalysis>();
  AU.addRequired<PostDominatorTree>();
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addRequired<LoopInfo>();
  AU.addRequired<ScalarEvolution>();
  AU.addRequired<NDRange>();
  AU.setPreservesAll();
}

//------------------------------------------------------------------------------
// Count all instruction types.
#define HANDLE_INST(N, OPCODE, CLASS)                                          \
  void OpenCLLoopFeatureExtractor::visit##OPCODE(CLASS &I) {                   \
    Instruction *inst = (Instruction *)&I;                                     \
    if (!isInLoop(inst, LI))                                                   \
      return;                                                                  \
    collector.instTypes[#OPCODE] += 1;                                         \
    collector.instTypes["insts"] += 1;                                         \
  }
#include "llvm/IR/Instruction.def"

//------------------------------------------------------------------------------
void OpenCLLoopFeatureExtractor::visitInstruction(Instruction &inst) {
  errs() << "Unknown instruction: " << inst;
  llvm_unreachable(0);
}

//------------------------------------------------------------------------------
void OpenCLLoopFeatureExtractor::visitBasicBlock(BasicBlock &basicBlock) {
  BasicBlock *block = (BasicBlock *)&basicBlock;
  
  if(!isInLoop(block, LI))
    return;

  collector.instTypes["blocks"] += 1;
  collector.computeILP(block);
  collector.computeMLP(block);
  collector.countInstsBlock(basicBlock);
  collector.countConstants(basicBlock);
  collector.countBarriers(basicBlock);
  collector.countMathFunctions(basicBlock);
  collector.countOutgoingEdges(basicBlock);
  collector.countIncomingEdges(basicBlock);
  collector.countLocalMemoryUsage(basicBlock);
  collector.countPhis(basicBlock);
  collector.livenessAnalysis(basicBlock);
}

//------------------------------------------------------------------------------
void OpenCLLoopFeatureExtractor::visitFunction(Function &function) {
  // Extract ThreadId values. 
//  InstVector tmp = SDDA->getThreadIds();
//  TIds = ToValueVector(tmp);
  collector.loopCountBranches(function, LI);
  collector.loopCountEdges(function, LI);
  collector.loopCountDivInsts(function, MDDA, SDDA, LI);
}
