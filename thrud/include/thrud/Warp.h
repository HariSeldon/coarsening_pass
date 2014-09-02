#ifndef WARP_H
#define WARP_H

#include "thrud/NDRangeSpace.h"
#include "thrud/NDRangePoint.h"

class NDRangeSpace;

class Warp {
public:
  // localGroupIndex: index of the local work group to which the warp belongs
  // to.
  // warpIndex: index of the warp in the local group.
  // Both are counted starting from 0 in raw major order.
  Warp(int groupX, int groupY, int groupZ, int warpIndex,
       const NDRangeSpace &ndrSpace);

private:
  std::vector<NDRangePoint> points;

  //-----------------------------------------------------------------------------
public:
  class iterator
      : public std::iterator<std::forward_iterator_tag, NDRangePoint> {
  public:
    iterator();
    iterator(const Warp *warp);
    iterator(const iterator &original);

  private:
    std::vector<NDRangePoint> points;
    size_t currentPoint;

  public:
    // Pre-increment.
    iterator &operator++();
    // Post-increment.
    iterator operator++(int);
    NDRangePoint operator*() const;
    bool operator!=(const iterator &iter) const;
    bool operator==(const iterator &iter) const;

    static iterator end();

  private:
    void toNext();
  };

public:
  Warp::iterator begin();
  Warp::iterator end();
};

#endif
