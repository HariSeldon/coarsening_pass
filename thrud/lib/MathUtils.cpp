#include "thrud/MathUtils.h"

#include <algorithm>
#include <functional>
#include <numeric>

#include <iostream>
#include <iterator>

// "Private" support function.
float square(float input) { return input * input; }

//------------------------------------------------------------------------------
template <typename integerType>
float getAverage(const std::vector<integerType> &elements) {
  if (elements.size() == 0)
    return 0;

  integerType sum = std::accumulate(elements.begin(), elements.end(), 0);
  float average = (float) sum / elements.size();

  return average;
}

template float getAverage(const std::vector<int> &elements);

//------------------------------------------------------------------------------
template <typename integerType>
float getVariance(const std::vector<integerType> &elements) {
  if (elements.size() == 0)
    return 0;

  float average = getAverage(elements);
  std::vector<float> averages(elements.size(), average);
  std::vector<float> differences;
  differences.reserve(elements.size());

  std::transform(elements.begin(), elements.end(), averages.begin(),
                 differences.begin(), std::minus<float>());

  std::transform(differences.begin(), differences.end(), differences.begin(),
                 square);

  float variance = getAverage(differences);

  return variance;
}

template float getVariance(const std::vector<unsigned int> &elements);
