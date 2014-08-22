#ifndef NDRANGE_SPACE_H
#define NDRANGE_SPACE_H

#include <string>
#include <vector>

class NDRangeSpace {
public:
  NDRangeSpace(int localSizeX, int localSizeY, int localSizeZ,
               int numberOfGroupsX, int numberOfGroupsY, int numberOfGroupsZ);

public:
  int getLocalSizeX() const;
  int getLocalSizeY() const;
  int getLocalSizeZ() const;

  int getGlobalSizeX() const;
  int getGlobalSizeY() const;
  int getGlobalSizeZ() const;

  int getNumberOfGroupsX() const;
  int getNumberOfGroupsY() const;
  int getNumberOfGroupsZ() const;

  int getGroupSize() const;

  int getLocalSize(int direction) const;
  int getGlobalSize(int direction) const;
  int getNumberOfGroups(int direction) const;

  int getSize(const std::string &name, int direction) const;

private:
  std::vector<int> localSize;
  std::vector<int> numberOfGroups;
};

#endif
