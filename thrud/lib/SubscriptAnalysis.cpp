#include "thrud/SubscriptAnalysis.h"

#include "thrud/NDRange.h"
#include "thrud/NDRangePoint.h"
#include "thrud/OCLEnv.h"
#include "thrud/Warp.h"

#include "llvm/IR/Instructions.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <functional>
#include <iterator>

int getTypeWidth(const Type *type);

//------------------------------------------------------------------------------
SubscriptAnalysis::SubscriptAnalysis(ScalarEvolution *scalarEvolution,
                                     OCLEnv *ocl, const Warp &warp)
    : scalarEvolution(scalarEvolution), ocl(ocl), warp(warp) {}

//------------------------------------------------------------------------------
int SubscriptAnalysis::getBankConflictNumber(Value *value) {
  //errs() << "SubscriptAnalysis::getBankConflictNumber\n";
  if (!isa<GetElementPtrInst>(value)) {
    return 0;
  }

  if (!scalarEvolution->isSCEVable(value->getType())) {
    return -1;
  }

  const SCEV *scev = scalarEvolution->getSCEV(value);
  std::vector<const SCEV *> addressesVector = analyzeSubscript(scev);
  return computeBankConflictNumber(addressesVector);
}

//------------------------------------------------------------------------------
// It might not cover all the cases. For the moment is good enough.
bool SubscriptAnalysis::isConsecutive(Value *value, int direction) {
  assert((direction == 0 || direction == 1 || direction == 2) &&
         "Unsupported direction");

  if (!isa<GetElementPtrInst>(value)) {
    return 0;
  }

  if (!scalarEvolution->isSCEVable(value->getType())) {
    return -1;
  }

  const SCEV *scev = scalarEvolution->getSCEV(value);

  const int TEST_NUMBER = 64;
  std::vector<const SCEV *> addresses;
  addresses.reserve(OCLEnv::WARP_SIZE);
  
  NDRangeSpace ndrSpace(1024, 1024, 1024, 1024, 1024, 1024);

  bool isFirst = direction == 0;
  bool isSecond = direction == 1;
  bool isThird = direction == 2;

  // Increment along direction. 
  for (int index = 0; index < TEST_NUMBER; ++index) {
    NDRangePoint point(isFirst * index, isSecond * index, isThird * index, 0, 0, 0, ndrSpace);
    SCEVMap processed;
    const SCEV *expr = replaceInExpr(scev, point, processed);
    addresses.push_back(expr);
  }

  assert((int)addresses.size() == TEST_NUMBER && "Missing addresses");

  const SCEV *first = addresses[0];
  int typeWidth = getTypeWidth(first->getType());

  const SCEV *unknown = getUnknownSCEV(first);
  if (unknown == nullptr) {
    return OCLEnv::WARP_SIZE;
  }

  verifyUnknown(addresses, unknown);

  std::vector<int> indices = getMemoryOffsets(addresses, unknown);

  // If any of the indices is UNKNOWN_MEMORY_LOCATION do something special.
  std::vector<int>::iterator unknownMemoryLocationPosition = std::find(
      indices.begin(), indices.end(), OCLEnv::UNKNOWN_MEMORY_LOCATION);

  if (unknownMemoryLocationPosition != indices.end()) {
    return false;
  }

  for (int index = 0; index < (int)(indices.size() - 1); ++index) {
    if(indices[index] + typeWidth != indices[index + 1])
      return false;
  }

  return true;
}

//------------------------------------------------------------------------------
int SubscriptAnalysis::getTransactionNumber(Value *value) {
  if (!isa<GetElementPtrInst>(value)) {
    return 0;
  }

  if (!scalarEvolution->isSCEVable(value->getType())) {
    return -1;
  }

  const SCEV *scev = scalarEvolution->getSCEV(value);
  std::vector<const SCEV *> addressesVector = analyzeSubscript(scev);
  return computeTransactionNumber(addressesVector);
}

//------------------------------------------------------------------------------
std::vector<const SCEV *>
SubscriptAnalysis::analyzeSubscript(const SCEV *scev) {
  std::vector<const SCEV *> resultVector;
  resultVector.reserve(OCLEnv::WARP_SIZE);

  // Iterate over threads in a warp.
  for (Warp::iterator iter = warp.begin(), iterEnd = warp.end();
       iter != iterEnd; ++iter) {
    NDRangePoint point = *iter;
    SCEVMap processed;
    const SCEV *expr = replaceInExpr(scev, point, processed);
    resultVector.push_back(expr);
  }

  assert((int)resultVector.size() == OCLEnv::WARP_SIZE &&
         "Missing expressions");
  return resultVector;
}

//------------------------------------------------------------------------------
int SubscriptAnalysis::computeBankConflictNumber(
    const std::vector<const SCEV *> &scevs) {

  const SCEV *first = scevs[0];
  const SCEV *unknown = getUnknownSCEV(first);
  if (unknown == nullptr) {
    // FIXME It don't know how to deal with this case. Yet.
    assert(false);
    return OCLEnv::BANK_NUMBER;
  }

  verifyUnknown(scevs, unknown);
  assert((int)scevs.size() == OCLEnv::WARP_SIZE && "Wrong number of SCEVs");

  std::vector<int> indices = getMemoryOffsets(scevs, unknown);

  // If any of the indices is UNKNOWN_MEMORY_LOCATION do something special.
  std::vector<int>::iterator unknownMemoryLocationPosition = std::find(
      indices.begin(), indices.end(), OCLEnv::UNKNOWN_MEMORY_LOCATION);

  if (unknownMemoryLocationPosition != indices.end()) {
    // FIXME It don't know how to deal with this case. Yet.
    assert(false);
    return OCLEnv::BANK_NUMBER;
  }

  std::vector<int> rows;
  std::vector<int> columns;

  rows.reserve(OCLEnv::WARP_SIZE);
  columns.reserve(OCLEnv::WARP_SIZE);

  const int LOCAL_MEMORY_WIDTH = OCLEnv::BANK_NUMBER * OCLEnv::BANK_WIDTH;

  // This is the actual computation of the number of bank conflicts.
  for (std::vector<int>::iterator iter = indices.begin(),
                                  iterEnd = indices.end();
       iter != iterEnd; ++iter) {
    columns.push_back(*iter % (OCLEnv::BANK_NUMBER));
  }

  std::transform(indices.begin(), indices.end(), std::back_inserter(rows),
                 std::bind2nd(std::divides<int>(), LOCAL_MEMORY_WIDTH));

  assert((int)rows.size() == OCLEnv::WARP_SIZE && "Wrong number of rows");
  assert(rows.size() == columns.size() && "Rows and Columns don't match");

  std::map<int, std::vector<int> > localMemory;

  for (int index = 0; index < OCLEnv::WARP_SIZE; ++index) {
    int row = rows[index];
    int column = columns[index];

    localMemory[column].push_back(row); 
  }

  for (std::map<int, std::vector<int> >::iterator iter = localMemory.begin(),
                                                  iterEnd = localMemory.end();
       iter != iterEnd; ++iter) {
    errs() << iter->first << " -- ";
    dumpIntVector(iter->second);
    errs() << "\n";
  }

  int conflictNumber = 0;

  for (std::map<int, std::vector<int> >::iterator iter = localMemory.begin(),
                                                  iterEnd = localMemory.end();
       iter != iterEnd; ++iter) {

    std::vector<int> &currentRows = iter->second;
    std::sort(currentRows.begin(), currentRows.end());

    std::vector<int>::iterator uniqueEnd =
        std::unique(currentRows.begin(), currentRows.end());
  
    int uniqueRows = std::distance(currentRows.begin(), uniqueEnd);
    --uniqueRows;

    conflictNumber = std::max(conflictNumber, uniqueRows);

  }

  return conflictNumber;
}

//------------------------------------------------------------------------------
int SubscriptAnalysis::computeTransactionNumber(
    const std::vector<const SCEV *> &scevs) {
  const SCEV *first = scevs[0];

  const SCEV *unknown = getUnknownSCEV(first);
  if (unknown == nullptr) {
    return OCLEnv::WARP_SIZE;
  }

  verifyUnknown(scevs, unknown);

  assert((int)scevs.size() == OCLEnv::WARP_SIZE && "Wrong number of SCEVs");

  std::vector<int> indices = getMemoryOffsets(scevs, unknown);

  // If any of the indices is UNKNOWN_MEMORY_LOCATION do something special.
  std::vector<int>::iterator unknownMemoryLocationPosition = std::find(
      indices.begin(), indices.end(), OCLEnv::UNKNOWN_MEMORY_LOCATION);

  if (unknownMemoryLocationPosition != indices.end()) {
    return OCLEnv::WARP_SIZE;
  }

  // This is the actual computation of the number of transactions. 
  std::transform(indices.begin(), indices.end(), indices.begin(),
                 std::bind2nd(std::divides<int>(), OCLEnv::CACHELINE_SIZE));

  std::sort(indices.begin(), indices.end());
  std::vector<int>::iterator uniqueEnd =
      std::unique(indices.begin(), indices.end());

  int uniqueCacheLines = std::distance(indices.begin(), uniqueEnd);

  return uniqueCacheLines;
}

//------------------------------------------------------------------------------
const SCEVUnknown *SubscriptAnalysis::getUnknownSCEV(const SCEV *scev) {
  if (const SCEVUnknown *unknown = dyn_cast<SCEVUnknown>(scev)) {
    return unknown;
  }

  if (const SCEVAddExpr *add = dyn_cast<SCEVAddExpr>(scev)) {
    if (add->getNumOperands() != 2)
      return nullptr;

    if (isa<SCEVConstant>(add->getOperand(0)) &&
        isa<SCEVUnknown>(add->getOperand(1))) {
      return dyn_cast<SCEVUnknown>(add->getOperand(1));
    }

    if (isa<SCEVConstant>(add->getOperand(1)) &&
        isa<SCEVUnknown>(add->getOperand(0))) {
      return dyn_cast<SCEVUnknown>(add->getOperand(0));
    }

    return nullptr;
  }

  return nullptr;
}

bool SubscriptAnalysis::verifyUnknown(const SCEV *scev, const SCEV *unknown) {
  if (const SCEVAddExpr *add = dyn_cast<SCEVAddExpr>(scev)) {
    return std::find(add->op_begin(), add->op_end(), unknown) != add->op_end();
  }
  return false;
}

bool SubscriptAnalysis::verifyUnknown(const std::vector<const SCEV *> &scevs,
                                      const SCEV *unknown) {
  bool ok = true;
  for (std::vector<const SCEV *>::const_iterator iter = scevs.begin(),
                                                 iterEnd = scevs.end();
       iter != iterEnd; ++iter) {
    ok &= verifyUnknown(*iter, unknown);
  }
  return ok;
}

//------------------------------------------------------------------------------
const SCEV *SubscriptAnalysis::replaceInExpr(const SCEV *expr,
                                             const NDRangePoint &point,
                                             SCEVMap &processed) {
  SCEVMap::iterator iter = processed.find(expr);
  if (iter != processed.end()) {
    return processed[expr];
  }

  const SCEV *result = nullptr;

  // FIXME: This is ugly.
  if (const SCEVCommutativeExpr *tmp = dyn_cast<SCEVCommutativeExpr>(expr))
    result = replaceInExpr(tmp, point, processed);
  if (const SCEVConstant *tmp = dyn_cast<SCEVConstant>(expr))
    result = replaceInExpr(tmp, point, processed);
  if (const SCEVUnknown *tmp = dyn_cast<SCEVUnknown>(expr))
    result = replaceInExpr(tmp, point, processed);
  if (const SCEVUDivExpr *tmp = dyn_cast<SCEVUDivExpr>(expr))
    result = replaceInExpr(tmp, point, processed);
  if (const SCEVAddRecExpr *tmp = dyn_cast<SCEVAddRecExpr>(expr))
    result = replaceInExpr(tmp, point, processed);
  if (const SCEVCastExpr *tmp = dyn_cast<SCEVCastExpr>(expr))
    result = replaceInExpr(tmp, point, processed);

  processed[expr] = result;

  return result;
}

//------------------------------------------------------------------------------
const SCEV *SubscriptAnalysis::replaceInExpr(const SCEVAddRecExpr *expr,
                                             const NDRangePoint &point,
                                             SCEVMap &processed) {
  //expr->dump();
  const SCEV *start = expr->getStart();
  // const SCEV* step = addRecExpr->getStepRecurrence(*scalarEvolution);
  // Check that the step is independent of the TID. TODO.
  return replaceInExpr(start, point, processed);
}

//------------------------------------------------------------------------------
const SCEV *SubscriptAnalysis::replaceInExpr(const SCEVCommutativeExpr *expr,
                                             const NDRangePoint &point,
                                             SCEVMap &processed) {
  //llvm::errs() << "SCEVCommutativeExpr:";
  //expr->dump();
  SmallVector<const SCEV *, 8> operands;
  for (SCEVNAryExpr::op_iterator I = expr->op_begin(), E = expr->op_end();
       I != E; ++I) {
    const SCEV *NewOperand = replaceInExpr(*I, point, processed);
    if (isa<SCEVCouldNotCompute>(NewOperand))
      return scalarEvolution->getCouldNotCompute();
    operands.push_back(NewOperand);
  }
  const SCEV *result = nullptr;

  //for (SmallVector<const SCEV *, 8>::iterator iter = operands.begin(),
  //                                            iterEnd = operands.end();
  //     iter != iterEnd; ++iter) {
  //  (*iter)->dump();
  //}

  if (isa<SCEVAddExpr>(expr))
    result = scalarEvolution->getAddExpr(operands);
  if (isa<SCEVMulExpr>(expr))
    result = scalarEvolution->getMulExpr(operands);
  if (isa<SCEVSMaxExpr>(expr))
    result = scalarEvolution->getSMaxExpr(operands);
  if (isa<SCEVUMaxExpr>(expr))
    result = scalarEvolution->getUMaxExpr(operands);

  return result;
}

//------------------------------------------------------------------------------
const SCEV *SubscriptAnalysis::replaceInExpr(const SCEVConstant *expr,
                                             const NDRangePoint &, SCEVMap &) {
  return expr;
}

//------------------------------------------------------------------------------
const SCEV *SubscriptAnalysis::replaceInExpr(const SCEVUnknown *expr,
                                             const NDRangePoint &point,
                                             SCEVMap &processed) {
  Value *value = expr->getValue();
  // Implement actual replacement.
  if (Instruction *instruction = dyn_cast<Instruction>(value)) {

    // Manage binary operations.
    if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(instruction)) {
      //llvm::errs() << "BinaryOperator: ";
      //BinOp->dump();

      // Modulo.
      if (BinOp->getOpcode() == Instruction::URem) {
        const SCEV *Arg = scalarEvolution->getSCEV(BinOp->getOperand(0));
        const SCEV *Modulo = scalarEvolution->getSCEV(BinOp->getOperand(1));
        const SCEV *Result = scalarEvolution->getMinusSCEV(
            Arg, scalarEvolution->getMulExpr(
                     scalarEvolution->getUDivExpr(Arg, Modulo), Modulo));
        return replaceInExpr(Result, point, processed);
      }

      // Signed division.
      if (BinOp->getOpcode() == Instruction::SDiv) {
        const SCEV *First = scalarEvolution->getSCEV(BinOp->getOperand(0));
        const SCEV *Second = scalarEvolution->getSCEV(BinOp->getOperand(1));
        const SCEV *Div = scalarEvolution->getUDivExpr(First, Second);
        return replaceInExpr(Div, point, processed);
      }

      //llvm::errs() << "Could not compute!\n";
      // All the rest.
      return scalarEvolution->getCouldNotCompute();
    }

    // Manage casts.
    if (isIntCast(instruction)) {
      CallInst *Call = dyn_cast<CallInst>(instruction);
      const SCEV *ArgSCEV = scalarEvolution->getSCEV(Call->getArgOperand(0));
      return replaceInExpr(ArgSCEV, point, processed);
    }

    // Manage phi nodes.
    if (PHINode *phi = dyn_cast<PHINode>(value))
      return replaceInPhi(phi, point, processed);

    return resolveInstruction(instruction, point);
  }

  // If the value is a function argument query OCL.
  if (isa<Argument>(value) && value->getType()->isIntegerTy())
    return scalarEvolution->getConstant(APInt(32, ocl->resolveValue(value)));

  return expr;
}

//------------------------------------------------------------------------------
const SCEV *
SubscriptAnalysis::resolveInstruction(llvm::Instruction *instruction,
                                      const NDRangePoint &point) {
  const NDRange *ndr = ocl->getNDRange();
  const NDRangeSpace &ndrSpace = ocl->getNDRangeSpace();
  std::string type = ndr->getType(instruction);

  // Check if the instruction is a coordinate querying the NDRangePoint class.
  if (ndr->isCoordinate(instruction)) {
    int direction = ndr->getDirection(instruction);
    int coordinate = point.getCoordinate(type, direction);
    return scalarEvolution->getConstant(APInt(32, coordinate));
  }

  // Check if the instruction is a size querying the NDRange class.
  if (ndr->isSize(instruction)) {
    int direction = ndr->getDirection(instruction);
    int size = ndrSpace.getSize(type, direction);
    return scalarEvolution->getConstant(APInt(32, size));
  }

  // If the instruction is neither a coordinate nor a size return
  // CouldNotCompute.
  return scalarEvolution->getCouldNotCompute();
}

//------------------------------------------------------------------------------
const SCEV *SubscriptAnalysis::replaceInExpr(const SCEVUDivExpr *expr,
                                             const NDRangePoint &point,
                                             SCEVMap &processed) {

  //  llvm::errs() << "SCEVUDiv: ";
  //  Expr->dump();
  const SCEV *newLHS = replaceInExpr(expr->getLHS(), point, processed);
  if (isa<SCEVCouldNotCompute>(newLHS))
    return scalarEvolution->getCouldNotCompute();
  const SCEV *newRHS = replaceInExpr(expr->getRHS(), point, processed);
  if (isa<SCEVCouldNotCompute>(newRHS))
    return scalarEvolution->getCouldNotCompute();

  return scalarEvolution->getUDivExpr(newLHS, newRHS);
}

//------------------------------------------------------------------------------
const SCEV *SubscriptAnalysis::replaceInExpr(const SCEVCastExpr *expr,
                                             const NDRangePoint &point,
                                             SCEVMap &processed) {
  return replaceInExpr(expr->getOperand(), point, processed);
}

//------------------------------------------------------------------------------
const SCEV *SubscriptAnalysis::replaceInPhi(PHINode *Phi,
                                            const NDRangePoint &point,
                                            SCEVMap &processed) {
  //llvm::errs() << "Phi: ";
  //Phi->dump();
  // FIXME: Pick the first argument of the phi node.
  Value *param = Phi->getIncomingValue(0);
  assert(scalarEvolution->isSCEVable(param->getType()) &&
         "PhiNode argument non-SCEVable");

  const SCEV *scev = scalarEvolution->getSCEV(param);

  processed[scalarEvolution->getSCEV(Phi)] = scev;

  return replaceInExpr(scev, point, processed);
}

//------------------------------------------------------------------------------
std::vector<int>
SubscriptAnalysis::getMemoryOffsets(std::vector<const SCEV *> scevs,
                                    const SCEV *unknown) {
  std::vector<const SCEV *> offsets;
  for (std::vector<const SCEV *>::const_iterator iter = scevs.begin(),
                                                 iterEnd = scevs.end();
       iter != iterEnd; ++iter) {
    const SCEV *currentSCEV = *iter;
    offsets.push_back(scalarEvolution->getMinusSCEV(currentSCEV, unknown));
  }

  assert(offsets.size() == scevs.size() && "Wrong number of offsets");

  std::vector<int> indices;
  // This is a std::transform, again.
  for (std::vector<const SCEV *>::const_iterator iter = offsets.begin(),
                                                 iterEnd = offsets.end();
       iter != iterEnd; ++iter) {

    const SCEV *currentSCEV = *iter;

    if (const SCEVAddRecExpr *AddRecSCEV =
            dyn_cast<SCEVAddRecExpr>(currentSCEV)) {
      currentSCEV = AddRecSCEV->getStart();
    }

    if (const SCEVConstant *ConstSCEV = dyn_cast<SCEVConstant>(currentSCEV)) {
      const ConstantInt *value = ConstSCEV->getValue();
      indices.push_back((int)value->getValue().roundToDouble());
    } else {
      // The SCEV is not constant. I don't know which element is accessed.
      indices.push_back(OCLEnv::UNKNOWN_MEMORY_LOCATION);
    }
  }
  
//  for (std::vector<int>::iterator iter = indices.begin(), iterEnd = indices.end(); iter != iterEnd; ++iter) {
//    errs() << *iter << "\n";
//  }

  assert(indices.size() == scevs.size() && "Wrong number of indices");
  return indices;
}

//------------------------------------------------------------------------------
int getTypeWidth(const Type *type) {
  assert(type->isPointerTy() && "Type is not a pointer");
  const Type *pointedType = type->getPointerElementType();
  int result = pointedType->getPrimitiveSizeInBits();
  if (result == 0) {
    return 32;
  }
  return result / 8;
}
