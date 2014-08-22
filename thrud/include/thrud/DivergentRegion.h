#ifndef DIVERGENT_REGION_H
#define DIVERGENT_REGION_H

#include "thrud/DataTypes.h"
#include "thrud/RegionBounds.h"

#include "llvm/Analysis/PostDominators.h"

#include "llvm/IR/Dominators.h"

namespace llvm {
class CmpInst;
class LoopInfo;
class ScalarEvolution;
}

class DivergentRegion {
public:
  DivergentRegion(BasicBlock *header, BasicBlock *exiting);
  DivergentRegion(BasicBlock *header, BasicBlock *exiting, InstVector &alive);
  DivergentRegion(RegionBounds &bounds);

  // Getter and Setter.
  BasicBlock *getHeader();
  BasicBlock *getExiting();

  const BasicBlock *getHeader() const;
  const BasicBlock *getExiting() const;

  RegionBounds &getBounds();
  BlockVector &getBlocks();
  InstVector &getAlive();
  InstVector &getIncoming();

  void setHeader(BasicBlock *Header);
  void setExiting(BasicBlock *Exiting);
  void setAlive(const InstVector &alive);
  void setIncoming(const InstVector &incoming);

  void fillRegion();
  void findAliveValues();
  void findIncomingValues();

  void analyze();
  //bool isStrict();
  bool areSubregionsDisjoint();

  DivergentRegion *clone(const Twine &suffix, DominatorTree *dt, Map &valueMap);
  BasicBlock *getSubregionExiting(unsigned int branchIndex);

  unsigned int size();
  void dump();
 

private:
  void updateBounds(DominatorTree *dt, PostDominatorTree *pdt); 

private:
  RegionBounds bounds;
  BlockVector blocks;
  InstVector alive;
  InstVector incoming;

public:
  // Iterator class.
  //-----------------------------------------------------------------------------
  class iterator
      : public std::iterator<std::forward_iterator_tag, BasicBlock *> {
  public:
    iterator();
    iterator(const DivergentRegion &region);
    iterator(const iterator &original);

  private:
    BlockVector blocks;
    size_t currentBlock;

  public:
    // Pre-increment.
    iterator &operator++();
    // Post-increment.
    iterator operator++(int);
    BasicBlock *operator*() const;
    bool operator!=(const iterator &iter) const;
    bool operator==(const iterator &iter) const;

    static iterator end();

  private:
    void toNext();
  };

  DivergentRegion::iterator begin();
  DivergentRegion::iterator end();

  // Const iterator class.
  //-----------------------------------------------------------------------------
  class const_iterator
      : public std::iterator<std::forward_iterator_tag, BasicBlock *> {
  public:
    const_iterator();
    const_iterator(const DivergentRegion &region);
    const_iterator(const const_iterator &original);

  private:
    BlockVector blocks;
    size_t currentBlock;

  public:
    // Pre-increment.
    const_iterator &operator++();
    // Post-increment.
    const_iterator operator++(int);
    const BasicBlock *operator*() const;
    bool operator!=(const const_iterator &iter) const;
    bool operator==(const const_iterator &iter) const;

    static const_iterator end();

  private:
    void toNext();
  };

  DivergentRegion::const_iterator begin() const;
  DivergentRegion::const_iterator end() const;
};

typedef std::vector<DivergentRegion *> RegionVector;

// Non member functions.
// -----------------------------------------------------------------------------
BasicBlock *getExit(DivergentRegion &region);
BasicBlock *getPredecessor(DivergentRegion *region, LoopInfo *loopInfo);
RegionBounds cloneRegion(RegionBounds &bounds, const Twine &suffix,
                         DominatorTree *dt);
bool contains(const DivergentRegion &region, const Instruction *inst);
bool containsInternally(const DivergentRegion &region, const Instruction *inst);
bool contains(const DivergentRegion &region, const BasicBlock *block);
bool containsInternally(const DivergentRegion &region, const BasicBlock *block);
bool containsInternally(const DivergentRegion &region,
                        const DivergentRegion *innerRegion);
BasicBlock *getSubregionExiting(DivergentRegion *region, unsigned int branchIndex);
void getSubregionAlive(DivergentRegion *region,
                       const BasicBlock *subregionExiting, InstVector &result);

#endif
