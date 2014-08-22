#include "thrud/DivergentRegion.h"

#include "thrud/RegionBounds.h"
#include "thrud/Utils.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ScalarEvolution.h"

#include "llvm/IR/Dominators.h"

#include "llvm/Support/raw_ostream.h"

#include "llvm/Transforms/Utils/Cloning.h"

#include <numeric>
#include <functional>
#include <algorithm>

DivergentRegion::DivergentRegion(BasicBlock *header, BasicBlock *exiting) { 
  bounds.setHeader(header);
  bounds.setExiting(exiting);

  fillRegion();
}

DivergentRegion::DivergentRegion(BasicBlock *header, BasicBlock *exiting,
                                 InstVector &alive)
    : alive(alive) {
  bounds.setHeader(header);
  bounds.setExiting(exiting);

  fillRegion();
}

DivergentRegion::DivergentRegion(RegionBounds &bounds)
    : bounds(bounds) {
  fillRegion();
}

BasicBlock *DivergentRegion::getHeader() { return bounds.getHeader(); }
BasicBlock *DivergentRegion::getExiting() { return bounds.getExiting(); }

const BasicBlock *DivergentRegion::getHeader() const {
  return bounds.getHeader();
}

const BasicBlock *DivergentRegion::getExiting() const {
  return bounds.getExiting();
}

void DivergentRegion::setHeader(BasicBlock *Header) {
  bounds.setHeader(Header);
}

void DivergentRegion::setExiting(BasicBlock *Exiting) {
  bounds.setExiting(Exiting);
}

void DivergentRegion::setAlive(const InstVector &alive) { this->alive = alive; }

void DivergentRegion::setIncoming(const InstVector &incoming) {
  this->incoming = incoming;
}

RegionBounds &DivergentRegion::getBounds() { return bounds; }

BlockVector &DivergentRegion::getBlocks() { return blocks; }

InstVector &DivergentRegion::getAlive() { return alive; }

InstVector &DivergentRegion::getIncoming() { return incoming; }

void DivergentRegion::fillRegion() {
  blocks.clear();
  bounds.listBlocks(blocks);
}

//------------------------------------------------------------------------------
void DivergentRegion::findAliveValues() {
  alive.clear();
  for (auto block : blocks) {
    for (auto iterInst = block->begin(), instEnd = block->end();
         iterInst != instEnd; ++iterInst) {
      Instruction *inst = iterInst;

      // Iterate over the uses of the instruction.
      for (Instruction::user_iterator iterUser = inst->user_begin(),
                                     userEnd = inst->user_end();
           iterUser != userEnd; ++iterUser) {
        if (Instruction *userInst = dyn_cast<Instruction>(*iterUser)) {
          BasicBlock *blockUser = userInst->getParent();
          // If the user of the instruction is not in the region -> the value is
          // alive.
          if (!contains(*this, blockUser)) {
            alive.push_back(inst);
            break;
          }
        }
      }
    }
  }
}

//------------------------------------------------------------------------------
void DivergentRegion::findIncomingValues() {
  incoming.clear();
  for (BlockVector::iterator iterBlock = blocks.begin(),
                             blockEnd = blocks.end();
       iterBlock != blockEnd; ++iterBlock) {
    BasicBlock *block = *iterBlock;
    for (BasicBlock::iterator iterInst = block->begin(), instEnd = block->end();
         iterInst != instEnd; ++iterInst) {
      Instruction *inst = iterInst;

      // Iterate over the operands of the instruction.
      for (unsigned int opIndex = 0, opEnd = inst->getNumOperands();
           opIndex != opEnd; ++opIndex) {
        if(Instruction *operand = dyn_cast<Instruction>(inst->getOperand(opIndex))) {
          if(!contains(*this, operand->getParent())) {
            incoming.push_back(operand);
          }
        }
      }
    }
  }
}

//------------------------------------------------------------------------------
void DivergentRegion::updateBounds(DominatorTree *dt, PostDominatorTree *pdt) {
  for (auto block : blocks) {
    if (dominatesAll(block, blocks, dt))
      bounds.setHeader(block);
    else if (postdominatesAll(block, blocks, pdt))
      bounds.setExiting(block);
  }
}

//------------------------------------------------------------------------------
bool DivergentRegion::areSubregionsDisjoint() {
  BranchInst *branch = dyn_cast<BranchInst>(getHeader()->getTerminator());
  assert(branch->getNumSuccessors() == 2 && "Wrong successor number");

  BasicBlock *first = branch->getSuccessor(0);
  BasicBlock *second = branch->getSuccessor(1);

  BlockVector firstList;
  BlockVector secondList;

  listBlocks(first, getExiting(), firstList);
  listBlocks(second, getExiting(), secondList);

  std::sort(firstList.begin(), firstList.end());
  std::sort(secondList.begin(), secondList.end());

  BlockVector intersection;
  std::set_intersection(firstList.begin(), firstList.end(), secondList.begin(),
                        secondList.end(), std::back_inserter(intersection));

  if (intersection.size() == 1) {
    return intersection[0] == getExiting();
  }
  return false;
}

//------------------------------------------------------------------------------
DivergentRegion *DivergentRegion::clone(const Twine &suffix, DominatorTree *dt,
                                        Map &valuesMap) {
  Function *function = getHeader()->getParent();
  BasicBlock *newHeader = nullptr;
  BasicBlock *newExiting = nullptr;

  Map regionBlockMap;
  valuesMap.clear();
  BlockVector newBlocks;
  newBlocks.reserve(blocks.size());
  for (auto block : blocks) {
    BasicBlock *newBlock =
        CloneBasicBlock(block, valuesMap, suffix, function, 0);
    regionBlockMap[block] = newBlock;
    newBlocks.push_back(newBlock);

    if (block == getHeader())
      newHeader = newBlock;
    if (block == getExiting())
      newExiting = newBlock;

    CloneDominatorInfo(block, regionBlockMap, dt);
  }

  // The remapping of the branches must be done at the end of the cloning
  // process.
  for (auto block : newBlocks) {
    applyMap(block, regionBlockMap);
    applyMap(block, valuesMap);
  }

  return new DivergentRegion(newHeader, newExiting);
}

//------------------------------------------------------------------------------
void DivergentRegion::dump() {
  errs() << "Bounds: " << getHeader()->getName() << " -- "
         << getExiting()->getName() << "\n";
  errs() << "Blocks: ";
  for (auto block : blocks) {
    errs() << block->getName() << ", ";
  }
  errs() << "\n";
}

//------------------------------------------------------------------------------
unsigned int addSizes(unsigned int partialSum, BasicBlock *block) {
  //  if (block == bounds.getHeader() || block == bounds.getExiting())
  //    return;
  return partialSum + block->size();
}

unsigned int DivergentRegion::size() {
  return std::accumulate(blocks.begin(), blocks.end(), 0, addSizes);
}

//------------------------------------------------------------------------------
DivergentRegion::iterator DivergentRegion::begin() { return iterator(*this); }

DivergentRegion::iterator DivergentRegion::end() {
  return DivergentRegion::iterator::end();
}

DivergentRegion::const_iterator DivergentRegion::begin() const {
  return const_iterator(*this);
}

DivergentRegion::const_iterator DivergentRegion::end() const {
  return DivergentRegion::const_iterator::end();
}

// Iterator class.
//------------------------------------------------------------------------------
DivergentRegion::iterator::iterator() { currentBlock = 0; }
DivergentRegion::iterator::iterator(const DivergentRegion &region) {
  blocks = region.blocks;
  currentBlock = (blocks.size() == 0) ? -1 : 0;
}
DivergentRegion::iterator::iterator(const iterator &original) {
  blocks = original.blocks;
  currentBlock = original.currentBlock;
}

// Pre-increment.
DivergentRegion::iterator &DivergentRegion::iterator::operator++() {
  toNext();
  return *this;
}
// Post-increment.
DivergentRegion::iterator DivergentRegion::iterator::operator++(int) {
  iterator old(*this);
  ++*this;
  return old;
}

BasicBlock *DivergentRegion::iterator::operator*() const {
  return blocks.at(currentBlock);
}
bool DivergentRegion::iterator::operator!=(const iterator &iter) const {
  return iter.currentBlock != this->currentBlock;
}

void DivergentRegion::iterator::toNext() {
  ++currentBlock;
  if (currentBlock == blocks.size())
    currentBlock = -1;
}

DivergentRegion::iterator DivergentRegion::iterator::end() {
  iterator endIterator;
  endIterator.currentBlock = -1;
  return endIterator;
}

// Const Iterator class.
//------------------------------------------------------------------------------
DivergentRegion::const_iterator::const_iterator() { currentBlock = 0; }
DivergentRegion::const_iterator::const_iterator(const DivergentRegion &region) {
  blocks = region.blocks;
  currentBlock = (blocks.size() == 0) ? -1 : 0;
}
DivergentRegion::const_iterator::const_iterator(
    const const_iterator &original) {
  blocks = original.blocks;
  currentBlock = original.currentBlock;
}

// Pre-increment.
DivergentRegion::const_iterator &DivergentRegion::const_iterator::operator++() {
  toNext();
  return *this;
}
// Post-increment.
DivergentRegion::const_iterator
DivergentRegion::const_iterator::operator++(int) {
  const_iterator old(*this);
  ++*this;
  return old;
}

const BasicBlock *DivergentRegion::const_iterator::operator*() const {
  return blocks.at(currentBlock);
}
bool
DivergentRegion::const_iterator::operator!=(const const_iterator &iter) const {
  return iter.currentBlock != this->currentBlock;
}

void DivergentRegion::const_iterator::toNext() {
  ++currentBlock;
  if (currentBlock == blocks.size())
    currentBlock = -1;
}

DivergentRegion::const_iterator DivergentRegion::const_iterator::end() {
  const_iterator endIterator;
  endIterator.currentBlock = -1;
  return endIterator;
}

// Non member functions.
//------------------------------------------------------------------------------
BasicBlock *getExit(DivergentRegion &region) {
  TerminatorInst *terminator = region.getExiting()->getTerminator();
  assert(terminator->getNumSuccessors() == 1 &&
         "Divergent region must have one successor only");
  BasicBlock *exit = terminator->getSuccessor(0);
  return exit;
}

//------------------------------------------------------------------------------
BasicBlock *getPredecessor(DivergentRegion *region, LoopInfo *loopInfo) {
  BasicBlock *header = region->getHeader();
  BasicBlock *predecessor = header->getSinglePredecessor();
  if (predecessor == nullptr) {
    Loop *loop = loopInfo->getLoopFor(header);
    predecessor = loop->getLoopPredecessor();
  }
  assert(predecessor != nullptr &&
         "Region header does not have a single predecessor");
  return predecessor;
}

// -----------------------------------------------------------------------------
bool contains(const DivergentRegion &region, const Instruction *inst) {
  return contains(region, inst->getParent());
}

bool containsInternally(const DivergentRegion &region,
                        const Instruction *inst) {
  return containsInternally(region, inst->getParent());
}

bool contains(const DivergentRegion &region, const BasicBlock *block) {
  auto result = std::find(region.begin(), region.end(), block);
  return result != region.end();
}

bool containsInternally(const DivergentRegion &region,
                        const BasicBlock *block) {
  for (auto currentBlock : region) {
    if (currentBlock == region.getHeader() ||
        currentBlock == region.getExiting())
      continue;

    if (block == currentBlock)
      return true;
  }
  return false;
}

//------------------------------------------------------------------------------
bool containsInternally(const DivergentRegion &region,
                        const DivergentRegion *innerRegion) {
  return containsInternally(region, innerRegion->getHeader()) &&
         containsInternally(region, innerRegion->getExiting());
}

BasicBlock *getSubregionExiting(DivergentRegion *region,
                                unsigned int branchIndex) {
  BasicBlock *exiting = region->getExiting();
  BranchInst *branch =
      dyn_cast<BranchInst>(region->getHeader()->getTerminator());
  assert(branch->getNumSuccessors() == 2 && "Wrong successor number");

  BasicBlock *top = branch->getSuccessor(branchIndex);
  BlockVector blocks;
  listBlocks(top, exiting, blocks);

  for (pred_iterator iter = pred_begin(exiting), iterEnd = pred_end(exiting);
       iter != iterEnd; ++iter) {
    BasicBlock *pred = *iter;
    if (isPresent(pred, blocks)) {
      return pred;
    }
  }
  return nullptr;
}

//------------------------------------------------------------------------------
void getSubregionAlive(DivergentRegion *region,
                       const BasicBlock *subregionExiting, InstVector &result) {
  // Identify the alive values defined in the merged subregion.
  InstVector &alive = region->getAlive();
  for (auto inst : alive) {
    if (PHINode *phi = dyn_cast<PHINode>(inst)) {
      result.push_back(dyn_cast<Instruction>(
          phi->getIncomingValueForBlock(subregionExiting)));
    } else {
      result.push_back(inst);
    }
  }
}
