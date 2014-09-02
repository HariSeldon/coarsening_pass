#include "thrud/Graph.h"

#include "thrud/Utils.h"

#include "llvm/IR/Instruction.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>

using namespace llvm;

//------------------------------------------------------------------------------
template <typename dataType>
Graph<dataType>::Graph(std::vector<dataType> &nodes)
    : nodes(nodes), edges(nodes.size() * nodes.size(), false) {}

//------------------------------------------------------------------------------
template <typename dataType> Graph<dataType>::~Graph() {}

//------------------------------------------------------------------------------
template <typename dataType>
void Graph<dataType>::addEdge(dataType source, dataType dest) {
  modifyGraph(source, dest, true);
}

//------------------------------------------------------------------------------
template <typename dataType>
void Graph<dataType>::removeEdge(dataType source, dataType dest) {
  modifyGraph(source, dest, false);
}

//------------------------------------------------------------------------------
template <typename dataType>
std::vector<dataType> indicesToElements(std::vector<dataType> &elements,
                                        std::vector<unsigned int> &indices) {
  // This would be replaced by copy_if in C++11.
  std::vector<dataType> result;
  for (std::vector<unsigned int>::iterator iter = indices.begin(),
                                           end = indices.end();
       iter != end; ++iter) {
    result.push_back(elements[*iter]);
  }

  return result;
}

//------------------------------------------------------------------------------
template <typename dataType>
std::vector<dataType> Graph<dataType>::getOutgoing(dataType source) {
  // Get all the indices on the row.
  unsigned int row = find(source);
  std::vector<unsigned int> columns;

  for (unsigned int column = 0; column < nodes.size(); ++column) {
    if (edges[row * nodes.size() + column])
      columns.push_back(column);
  }

  std::vector<dataType> result = indicesToElements(nodes, columns);
  return result;
}

//------------------------------------------------------------------------------
template <typename dataType>
std::vector<dataType> Graph<dataType>::getIncoming(dataType dest) {
  // Get all the indices on the column.
  unsigned int column = find(dest);
  std::vector<unsigned int> rows;

  for (unsigned int row = 0; row < nodes.size(); ++row) {
    if (edges[row * nodes.size() + column])
      rows.push_back(row);
  }

  std::vector<dataType> result = indicesToElements(nodes, rows);
  return result;
}

//------------------------------------------------------------------------------
template <typename dataType>
void Graph<dataType>::modifyGraph(dataType source, dataType dest,
                                  bool newValue) {
  unsigned int row = find(source);
  unsigned int column = find(dest);

  unsigned int position = row * nodes.size() + column;
  edges[position] = newValue;
}

//------------------------------------------------------------------------------
template <typename dataType>
unsigned int Graph<dataType>::find(dataType toFind) {
  typename std::vector<dataType>::iterator element =
      std::find(nodes.begin(), nodes.end(), toFind);
  unsigned int position = std::distance(nodes.begin(), element);
  return position;
}

template class Graph<Instruction *>;
