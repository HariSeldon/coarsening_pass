#include "thrud/NDRangeSpace.h"

#include "thrud/NDRange.h"

NDRangeSpace::NDRangeSpace(int localSizeX, int localSizeY, int localSizeZ,
                           int numberOfGroupsX, int numberOfGroupsY,
                           int numberOfGroupsZ) {
  int tmpLocalSize[] = {localSizeX, localSizeY, localSizeZ};
  int tmpNumberOfGroups[] = {numberOfGroupsX, numberOfGroupsY, numberOfGroupsZ};

  localSize.assign(tmpLocalSize, tmpLocalSize + NDRange::DIRECTION_NUMBER);
  numberOfGroups.assign(tmpNumberOfGroups,
                        tmpNumberOfGroups + NDRange::DIRECTION_NUMBER);
}

int NDRangeSpace::getLocalSizeX() const { return localSize[0]; }
int NDRangeSpace::getLocalSizeY() const { return localSize[1]; }
int NDRangeSpace::getLocalSizeZ() const { return localSize[2]; }

int NDRangeSpace::getGlobalSizeX() const { return getGlobalSize(0); }
int NDRangeSpace::getGlobalSizeY() const { return getGlobalSize(1); }
int NDRangeSpace::getGlobalSizeZ() const { return getGlobalSize(2); }

int NDRangeSpace::getNumberOfGroupsX() const { return numberOfGroups[0]; }
int NDRangeSpace::getNumberOfGroupsY() const { return numberOfGroups[1]; }
int NDRangeSpace::getNumberOfGroupsZ() const { return numberOfGroups[2]; }

int NDRangeSpace::getGroupSize() const {
  return localSize[0] * localSize[1] * localSize[2];
}

int NDRangeSpace::getLocalSize(int direction) const {
  return localSize[direction];
}

int NDRangeSpace::getGlobalSize(int direction) const {
  return localSize[direction] * numberOfGroups[direction];
}

int NDRangeSpace::getNumberOfGroups(int direction) const {
  return numberOfGroups[direction];
}

int NDRangeSpace::getSize(const std::string &name, int direction) const {
  if (name == NDRange::GET_LOCAL_SIZE)
    return getLocalSize(direction);
  if (name == NDRange::GET_GLOBAL_SIZE)
    return getGlobalSize(direction);
  if (name == NDRange::GET_GROUPS_NUMBER)
    return getNumberOfGroups(direction);

  return -1;
}
