#include "thrud/ILPComputation.h"

#include "thrud/DataTypes.h"
#include "thrud/Graph.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"

#include "llvm/Support/raw_ostream.h"

//------------------------------------------------------------------------------
void buildGraph(BasicBlock *block, Graph<Instruction *> &graph) {
  for (BasicBlock::iterator iter = block->begin(), end = block->end();
       iter != end; ++iter) {
    Instruction *inst = iter;

    std::vector<Instruction *> operands;
    // Get the instruction-operands belonging to the current block.
    for (Instruction::op_iterator opIter = inst->op_begin(),
                                  opEnd = inst->op_end();
         opIter != opEnd; ++opIter) {
      if (Instruction *opInst = dyn_cast<Instruction>(opIter)) {
        if (opInst->getParent() == block)
          graph.addEdge(inst, opInst);
      }
    }
  }
}

//------------------------------------------------------------------------------
InstVector getInsts(BasicBlock *block) {
  InstVector result;
  for (BasicBlock::iterator iter = block->begin(), end = block->end();
       iter != end; ++iter) {
    result.push_back(iter);
  }
  return result;
}

//------------------------------------------------------------------------------
std::vector<unsigned int>
getDephts(std::map<Instruction *, unsigned int> &inputMap, InstVector &insts) {

  std::vector<unsigned int> result;
  for (InstVector::iterator iter = insts.begin(), end = insts.end();
       iter != end; ++iter) {
    Instruction *inst = *iter;
    result.push_back(inputMap[inst]);
  }
  return result;
}

//------------------------------------------------------------------------------
float getILP(BasicBlock *block) {
  InstVector insts = getInsts(block);
  Graph<Instruction *> graph(insts);
  buildGraph(block, graph);

  std::map<Instruction *, unsigned int> depths;
  unsigned int maxDepth = 0;

  for (InstVector::iterator iter = insts.begin(), end = insts.end();
       iter != end; ++iter) {
    Instruction *inst = *iter;

    InstVector outgoing = graph.getOutgoing(inst);
    std::vector<unsigned int> currentDepths = getDephts(depths, outgoing);

    // Get the max depth of the adjacent nodes.
    std::vector<unsigned int>::iterator maxPosition =
        std::max_element(currentDepths.begin(), currentDepths.end());
    unsigned int max = 0;
    if (maxPosition != currentDepths.end()) {
      max = *maxPosition;
    }

    // Update the maxium depth reached so far.
    unsigned int currentDepth = max + 1;
    if (currentDepth > maxDepth)
      maxDepth = currentDepth;

    // Set the depth of the current instruction.
    depths[inst] = currentDepth;
  }

  float ilp = (float) insts.size() / maxDepth;
  return ilp;
}
