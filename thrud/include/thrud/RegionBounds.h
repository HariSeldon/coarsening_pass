#ifndef REGION_BOUNDS_H
#define REGION_BOUNDS_H

#include "thrud/DataTypes.h"

namespace llvm {
class BasicBlock;
}

class RegionBounds {
public:
  RegionBounds(BasicBlock *header, BasicBlock *exiting);
  RegionBounds();

public:
  BasicBlock *getHeader();
  BasicBlock *getExiting();
  const BasicBlock *getHeader() const;
  const BasicBlock *getExiting() const;

  void setHeader(BasicBlock *Header);
  void setExiting(BasicBlock *Exiting);

  void listBlocks(BlockVector &result);

  void dump(const std::string &prefix = "") const;

private:
  BasicBlock *header;
  BasicBlock *exiting;
};

void listBlocks(RegionBounds *bounds, BlockVector &result);
void listBlocks(BasicBlock *header, BasicBlock *exiting, BlockVector &result);

#endif
