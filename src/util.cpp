#include "util.h"

namespace Tensor {

int32_t arrayProduct(const std::vector<int32_t> &shape) {
  int32_t initialProduct = 1;
  return std::accumulate(shape.begin(), shape.end(), initialProduct,
                         std::multiplies<int32_t>());
}

std::vector<int32_t> getStride(const std::vector<int32_t> &shape) {
  int32_t size = arrayProduct(shape);
  std::vector<int32_t> stride(shape.size(), 0);
  int32_t stride_step = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    stride[i] = stride_step;
    stride_step *= shape[i];
  }
  return stride;
}


}