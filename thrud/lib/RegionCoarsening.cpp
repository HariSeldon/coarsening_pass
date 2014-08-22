#include "thrud/ThreadCoarsening.h"

#include "thrud/DataTypes.h"
#include "thrud/Utils.h"

#include "llvm/Analysis/LoopInfo.h"

#include "llvm/IR/Module.h"

#include "llvm/Transforms/Utils/Cloning.h"

//------------------------------------------------------------------------------
void ThreadCoarsening::replicateRegion(DivergentRegion *region) {
  assert(dt->dominates(region->getHeader(), region->getExiting()) &&
         "Header does not dominates Exiting");
  assert(pdt->dominates(region->getExiting(), region->getHeader()) &&
         "Exiting does not post dominate Header");

  replicateRegionClassic(region);
}

//------------------------------------------------------------------------------
void ThreadCoarsening::replicateRegionClassic(DivergentRegion *region) {
  CoarseningMap aliveMap;
  initAliveMap(region, aliveMap);
  replicateRegionImpl(region, aliveMap);
  updatePlaceholdersWithAlive(aliveMap);
}

//------------------------------------------------------------------------------
void ThreadCoarsening::initAliveMap(DivergentRegion *region,
                                    CoarseningMap &aliveMap) {
  InstVector &aliveInsts = region->getAlive();
  for (auto &inst : aliveInsts) {
    aliveMap.insert(std::pair<Instruction *, InstVector>(inst, InstVector()));
  }
}

//------------------------------------------------------------------------------
void ThreadCoarsening::replicateRegionImpl(DivergentRegion *region,
                                           CoarseningMap &aliveMap) {
  BasicBlock *pred = getPredecessor(region, loopInfo);
  BasicBlock *topInsertionPoint = region->getExiting();
  BasicBlock *bottomInsertionPoint = getExit(*region);

  // Replicate the region.
  for (unsigned int index = 0; index < factor - 1; ++index) {
    Map valueMap;
    DivergentRegion *newRegion =
        region->clone(".cf" + Twine(index + 2), dt, valueMap);
    applyCoarseningMap(*newRegion, index);

    // Connect the region to the CFG.
    changeBlockTarget(topInsertionPoint, newRegion->getHeader());
    changeBlockTarget(newRegion->getExiting(), bottomInsertionPoint);

    // Update the phi nodes of the newly inserted header.
    remapBlocksInPHIs(newRegion->getHeader(), pred, topInsertionPoint);
    // Update the phi nodes in the exit block.
    remapBlocksInPHIs(bottomInsertionPoint, topInsertionPoint,
                      newRegion->getExiting());

    topInsertionPoint = newRegion->getExiting();
    bottomInsertionPoint = getExit(*newRegion);

    delete newRegion;
    updateAliveMap(aliveMap, valueMap);
  }
}

//------------------------------------------------------------------------------
void ThreadCoarsening::updateAliveMap(CoarseningMap &aliveMap, Map &regionMap) {
  for (auto &mapIter : aliveMap) {
    InstVector &coarsenedInsts = mapIter.second;
    Value *value = regionMap[mapIter.first];
    assert(value != nullptr && "Missing alive value in region map");
    coarsenedInsts.push_back(dyn_cast<Instruction>(value));
  }
}

//------------------------------------------------------------------------------
void ThreadCoarsening::updatePlaceholdersWithAlive(CoarseningMap &aliveMap) {
  // Force the addition of the alive values to the coarsening map. 
  for (auto mapIter : aliveMap) {
    Instruction *alive = mapIter.first;
    InstVector &coarsenedInsts = mapIter.second;

    auto cIter = cMap.find(alive); 
    if(cIter == cMap.end()) {
      cMap.insert(std::pair<Instruction *, InstVector>(alive, coarsenedInsts)); 
    }
  }
  
  for (auto &mapIter : aliveMap) {
    Instruction *alive = mapIter.first;
    InstVector &coarsenedInsts = mapIter.second;

    updatePlaceholderMap(alive, coarsenedInsts);
  }
}
