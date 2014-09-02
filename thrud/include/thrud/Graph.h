#ifndef GRAPH_H
#define GRAPH_H

#include <bitset>
#include <vector>

// Implementation of a graph an adjacency matrix.
// True in row A, column B, means that there is an edge
// from node A to B.
template <typename dataType> class Graph {
public:
  Graph(std::vector<dataType> &nodes);
  ~Graph();

private:
  std::vector<dataType> nodes;
  // This could be represented with a std::bitset
  // but I prefer not to pass the total number of edges
  // as a template parameter since it can mismatch with the
  // length of the list of nodes.
  std::vector<bool> edges;

public:
  void addEdge(dataType source, dataType dest);
  void removeEdge(dataType source, dataType dest);
  std::vector<dataType> getOutgoing(dataType source);
  std::vector<dataType> getIncoming(dataType dest);

private:
  void modifyGraph(dataType source, dataType dest, bool newValue);
  unsigned int find(dataType toFind);
};

#endif
