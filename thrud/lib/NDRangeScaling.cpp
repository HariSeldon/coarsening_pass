#include "thrud/ThreadCoarsening.h"

#include "thrud/DataTypes.h"
#include "thrud/Utils.h"

// Support functions.
Instruction *getMulInst(Value *value, unsigned int factor);
Instruction *getAddInst(Value *value, unsigned int addend);
Instruction *getAddInst(Value *V1, Value *V2);
Instruction *getShiftInst(Value *value, unsigned int shift);
Instruction *getAndInst(Value *value, unsigned int factor);
Instruction *getDivInst(Value *value, unsigned int divisor);
Instruction *getModuloInst(Value *value, unsigned int modulo);

//------------------------------------------------------------------------------
void ThreadCoarsening::scaleNDRange() {
  InstVector InstTids;
  scaleSizes();
  scaleIds();
}

//------------------------------------------------------------------------------
void ThreadCoarsening::scaleSizes() {
  InstVector sizeInsts = ndr->getSizes(direction);
  for (InstVector::iterator iter = sizeInsts.begin(), iterEnd = sizeInsts.end();
       iter != iterEnd; ++iter) {
    // Scale size.
    Instruction *inst = *iter;
    Instruction *mul = getMulInst(inst, factor);
    mul->insertAfter(inst);
    // Replace uses of the old size with the scaled one.
    replaceUses(inst, mul);
  }
}

//------------------------------------------------------------------------------
// Scaling function: origTid = [newTid / st] * cf * st + newTid % st + subid * st
void ThreadCoarsening::scaleIds() {
  unsigned int cfst = factor * stride;

  InstVector tids = ndr->getTids(direction);
  for (InstVector::iterator instIter = tids.begin(), instEnd = tids.end();
       instIter != instEnd; ++instIter) {
    Instruction *inst = *instIter;

    // Compute base of new tid.
    Instruction *div = getDivInst(inst, stride); 
    div->insertAfter(inst);
    Instruction *mul = getMulInst(div, cfst);
    mul->insertAfter(div);
    Instruction *modulo = getModuloInst(inst, stride);
    modulo->insertAfter(mul);
    Instruction *base = getAddInst(mul, modulo);
    base->insertAfter(modulo);

    // Replace uses of the threadId with the new base.
    replaceUses(inst, base);
    modulo->setOperand(0, inst);
    div->setOperand(0, inst);

    // Compute the remaining thread ids.
    cMap.insert(std::pair<Instruction *, InstVector>(inst, InstVector()));
    InstVector &current = cMap[base];
    current.reserve(factor - 1);

    Instruction *bookmark = base;
    for (unsigned int index = 2; index <= factor; ++index) {
      Instruction *add = getAddInst(base, (index - 1) * stride);
      add->insertAfter(bookmark);
      current.push_back(add);
      bookmark = add;
    }
  }
}

// Support functions.
//------------------------------------------------------------------------------
unsigned int getIntWidth(Value *value) {
  Type *type = value->getType();
  IntegerType *intType = dyn_cast<IntegerType>(type);
  assert(intType && "Value type is not integer");
  return intType->getBitWidth();
}

ConstantInt *getConstantInt(unsigned int value, unsigned int width,
                            LLVMContext &context) {
  IntegerType *integer = IntegerType::get(context, width);
  return ConstantInt::get(integer, value);
}

Instruction *getMulInst(Value *value, unsigned int factor) {
  unsigned int width = getIntWidth(value);
  ConstantInt *factorValue = getConstantInt(factor, width, value->getContext());
  Instruction *mul =
      BinaryOperator::Create(Instruction::Mul, value, factorValue);
  mul->setName(value->getName() + ".." + Twine(factor));
  return mul;
}

Instruction *getAddInst(Value *value, unsigned int addend) {
  unsigned int width = getIntWidth(value);
  ConstantInt *addendValue = getConstantInt(addend, width, value->getContext());
  Instruction *add =
      BinaryOperator::Create(Instruction::Add, value, addendValue);
  add->setName(value->getName() + ".." + Twine(addend));
  return add;
}

Instruction *getAddInst(Value *firstValue, Value *secondValue) {
  Instruction *add =
      BinaryOperator::Create(Instruction::Add, firstValue, secondValue);
  add->setName(firstValue->getName() + "..Add");
  return add;
}

Instruction *getShiftInst(Value *value, unsigned int shift) {
  unsigned int width = getIntWidth(value);
  ConstantInt *intValue = getConstantInt(shift, width, value->getContext());
  Instruction *shiftInst =
      BinaryOperator::Create(Instruction::LShr, value, intValue);
  shiftInst->setName(Twine(value->getName()) + "..Shift");
  return shiftInst;
}

Instruction *getAndInst(Value *value, unsigned int factor) {
  unsigned int width = getIntWidth(value);
  ConstantInt *intValue = getConstantInt(factor, width, value->getContext());
  Instruction *andInst =
      BinaryOperator::Create(Instruction::And, value, intValue);
  andInst->setName(Twine(value->getName()) + "..And");
  return andInst;
}

Instruction *getDivInst(Value *value, unsigned int divisor) {
  unsigned int width = getIntWidth(value);
  ConstantInt *intValue = getConstantInt(divisor, width, value->getContext());
  Instruction *divInst = 
    BinaryOperator::Create(Instruction::UDiv, value, intValue);
  divInst->setName(Twine(value->getName()) + "..Div");
  return divInst;
}

Instruction *getModuloInst(Value *value, unsigned int modulo) {
  unsigned int width = getIntWidth(value);
  ConstantInt *intValue = getConstantInt(modulo, width, value->getContext());
  Instruction *moduloInst = 
    BinaryOperator::Create(Instruction::URem, value, intValue);
  moduloInst->setName(Twine(value->getName()) + "..Rem");
  return moduloInst;
}

