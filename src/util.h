#pragma once
#include <algorithm>
#include <numeric>
#include <vector>

namespace Tensor {
int32_t arrayProduct(const std::vector<int32_t> &shape);
std::vector<int32_t> getStride(const std::vector<int32_t> &shape);

template <typename T, typename E = std::enable_if_t<std::is_integral_v<T>>>
T CeilDiv(T a, T b) {
  return (a - 1) / b + 1;
}

#define assert(cond)                                                           \
  {                                                                            \
    if (cond) {                                                                \
    } else {                                                                   \
      throw;                                                                   \
    }                                                                          \
  }

}