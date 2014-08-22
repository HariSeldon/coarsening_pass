#include "thrud/ControlDependenceAnalysis.h"

#include "thrud/Utils.h"

#include "llvm/Analysis/CFG.h"

#include <algorithm>
#include <functional>

ControlDependenceAnalysis::ControlDependenceAnalysis() : FunctionPass(ID) {}

ControlDependenceAnalysis::~ControlDependenceAnalysis() {}

// -----------------------------------------------------------------------------
void ControlDependenceAnalysis::getAnalysisUsage(AnalysisUsage &au) const {
  au.addRequired<PostDominatorTree>();
  au.setPreservesAll();
}

// -----------------------------------------------------------------------------
bool ControlDependenceAnalysis::runOnFunction(Function &function) {
  pdt = &getAnalysis<PostDominatorTree>();

  forwardGraph.clear();
  backwardGraph.clear();
  s.clear();
  ls.clear();

  findS(function);
  findLs();
  buildGraph();
  fillGraph(function);
  transitiveClosure();
  buildBackwardGraph();

  return false;
}

// -----------------------------------------------------------------------------
void ControlDependenceAnalysis::findS(Function &function) {
  for (Function::iterator iter = function.begin(), iterEnd = function.end();
       iter != iterEnd; ++iter) {
    BasicBlock *block = iter;
    for (succ_iterator succIter = succ_begin(block), succEnd = succ_end(block);
         succIter != succEnd; ++succIter) {
      BasicBlock *child = *succIter;
      if (!pdt->dominates(child, block)) {
        s.push_back(std::pair<BasicBlock *, BasicBlock *>(block, child));
      }
    }
  }
}

// -----------------------------------------------------------------------------
void ControlDependenceAnalysis::findLs() {
  for (auto edge : s) {
    BasicBlock *block =
        pdt->findNearestCommonDominator(edge.first, edge.second);
    assert(block != nullptr && "Ill-formatted function");
    ls.push_back(block);
  }
  assert(ls.size() == s.size() && "Mismatching S and Ls");
}

// -----------------------------------------------------------------------------
void ControlDependenceAnalysis::buildGraph() {
  unsigned int edgeNumber = s.size();
  for (unsigned int index = 0; index < edgeNumber; ++index) {
    std::pair<BasicBlock *, BasicBlock *> edge = s[index];
    BasicBlock *l = ls[index];

    BasicBlock *a = edge.first;
    BasicBlock *b = edge.second;

    BlockVector children;
    BasicBlock *aParent = pdt->getNode(a)->getIDom()->getBlock();

    // Case 1.
    if (l == aParent) {
      BasicBlock *current = b;
      while (current != l) {
        children.push_back(current);
        current = pdt->getNode(current)->getIDom()->getBlock();
      }
    }

    // Case 2.
    if (l == a) {
      BasicBlock *current = b;
      while (current != aParent) {
        children.push_back(current);
        current = pdt->getNode(current)->getIDom()->getBlock();
      }
    }
    forwardGraph.insert(std::pair<BasicBlock *, BlockVector>(a, children));
  }
}

// -----------------------------------------------------------------------------
void ControlDependenceAnalysis::fillGraph(Function &function) {
  for (Function::iterator iter = function.begin(), iterEnd = function.end();
       iter != iterEnd; ++iter) {
    BasicBlock *block = iter;
    forwardGraph[block];
  }
}

// -----------------------------------------------------------------------------
void ControlDependenceAnalysis::transitiveClosure() {
  for (auto graphIter : forwardGraph) {
    BlockVector &seeds = graphIter.second;
    // This implements a traversal of the tree starting from block.
    BlockSet worklist(seeds.begin(), seeds.end());
    BlockVector result;
    while (!worklist.empty()) {
      BlockSet::iterator iter = worklist.begin();
      BasicBlock *current = *iter;
      worklist.erase(iter);
      result.push_back(current);
      BlockVector &children = forwardGraph[current];

      for (auto block : children) {
        if (!isPresent(block, result))
          worklist.insert(block);
      }
    }

    // Update block vector.
    graphIter.second.assign(result.begin(), result.end());
  }
}

// -----------------------------------------------------------------------------
void ControlDependenceAnalysis::buildBackwardGraph() {
  for (auto graphIter : forwardGraph) {
    BasicBlock *block = graphIter.first;
    backwardGraph.insert(
        std::pair<BasicBlock *, BlockVector>(block, BlockVector()));
    BlockVector &result = backwardGraph[block];

    for (auto graphIter2 : forwardGraph) {
      if (graphIter.first == graphIter2.first)
        continue;
      BlockVector &children = graphIter2.second;

      if (isPresent(block, children)) {
        result.push_back(graphIter2.first);
      }
    }
  }
}

// Public functions.
// -----------------------------------------------------------------------------
bool ControlDependenceAnalysis::dependsOn(BasicBlock *first,
                                          BasicBlock *second) {
  BlockVector &blocks = backwardGraph[first];
  return isPresent(second, blocks);
}

// -----------------------------------------------------------------------------
bool ControlDependenceAnalysis::dependsOn(Instruction *first,
                                          Instruction *second) {
  return dependsOn(first->getParent(), second->getParent());
}

// -----------------------------------------------------------------------------
bool ControlDependenceAnalysis::dependsOnAny(BasicBlock *inputBlock,
                                             BlockVector &blocks) {
  bool result =
      std::accumulate(blocks.begin(), blocks.end(), false,
                      [inputBlock, this](bool result, BasicBlock *block) {
        return result | dependsOn(inputBlock, block);
      });

  return result;
}

// -----------------------------------------------------------------------------
bool ControlDependenceAnalysis::dependsOnAny(Instruction *inst,
                                             InstVector &insts) {
  BlockVector blocks;
  blocks.resize(insts.size());
  std::transform(insts.begin(), insts.end(), blocks.begin(),
                 [](Instruction *inst) { return inst->getParent(); });

  return dependsOnAny(inst->getParent(), blocks);
}

// -----------------------------------------------------------------------------
bool ControlDependenceAnalysis::controls(BasicBlock *first,
                                         BasicBlock *second) {
  BlockVector &blocks = forwardGraph[first];
  return isPresent(second, blocks);
}

// -----------------------------------------------------------------------------
void ControlDependenceAnalysis::dump() {
  errs() << "Forward:\n";
  for (auto graphIter : forwardGraph) {
    BasicBlock *block = graphIter.first;
    errs() << block->getName() << ": ";
    BlockVector &children = graphIter.second;
    for (auto child : children) {
      errs() << child->getName() << " ";
    }
    errs() << "\n";
  }
  errs() << "Backward:\n";
  for (auto graphIter : backwardGraph) {
    BasicBlock *block = graphIter.first;
    errs() << block->getName() << ": ";
    BlockVector &children = graphIter.second;
    for (auto child : children) {
      errs() << child->getName() << " ";
    }
    errs() << "\n";
  }
}

char ControlDependenceAnalysis::ID = 0;
static RegisterPass<ControlDependenceAnalysis> X("dependence-analysis",
                                                 "Control dependence analysis");
