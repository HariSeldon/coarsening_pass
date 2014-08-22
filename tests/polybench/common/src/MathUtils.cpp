#include <math.h>
#include <stdlib.h>

#define SMALL_FLOAT_VAL 0.0000000001

//-----------------------------------------------------------------------------
template <typename type> type random() {
  type result = (type)rand() / (type)RAND_MAX;
  return (type)((1.0 - result) * 0 + result * 10.0);
}

template float random();
template double random();
template int random();

//-----------------------------------------------------------------------------
float percentDiff(double val1, double val2) {
  if ((fabs(val1) < 0.01) && (fabs(val2) < 0.01)) {
    return 0.0f;
  } else {
    return 100.0f *
           (fabs(fabs(val1 - val2) / fabs(val1 + SMALL_FLOAT_VAL)));
  }
}
