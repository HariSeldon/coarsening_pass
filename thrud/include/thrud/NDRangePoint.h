#ifndef NDRANGE_POINT_H
#define NDRANGE_POINT_H

#include <string>
#include <vector>

class NDRangeSpace;

class NDRangePoint {
public:
  NDRangePoint(int localX, int localY, int localZ, int groupX, int groupY,
               int groupZ, const NDRangeSpace &ndRangeSpace);

  int getLocalX() const;
  int getLocalY() const;
  int getLocalZ() const;

  int getGlobalX() const;
  int getGlobalY() const;
  int getGlobalZ() const;

  int getGroupX() const;
  int getGroupY() const;
  int getGroupZ() const;

  int getLocal(int direction) const;
  int getGlobal(int direction) const;
  int getGroup(int direction) const;

  int getCoordinate(const std::string &name, int direction) const;

  std::string toString() const;

private:
  std::vector<int> local;
  std::vector<int> global;
  std::vector<int> group;
};

#endif
