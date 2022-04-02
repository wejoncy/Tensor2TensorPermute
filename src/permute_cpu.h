#pragma once
#include "permute.h"
#include <algorithm>
#include <cstddef>
#include <ctype.h>


namespace Tensor {

//A implementation for any tensor permute which performed in CPU
class PermuteCPU : public PermuteBase {
private:
  // Given linear_index, to calculate the tensor index,
  std::vector<int32_t> linear2TensorIndex(const std::vector<int> &stride,
                                          int32_t linear_index) {
    std::vector<int32_t> tensor_index(stride.size(), 0);

    for (int32_t dim_index = stride.size() - 1;
         dim_index >= 0 && linear_index > 0; --dim_index) {
      if (dim_index == 0) {
        tensor_index[dim_index] = linear_index;
        continue;
      }
      tensor_index[dim_index] = linear_index % stride[dim_index];
      linear_index /= stride[dim_index];
    }
    return tensor_index;
  }
  // Given tensor index, to calculate the linear index. But please note that
  // some linear-index is out-of-bound so we have to return a invalid
  // index-value. the reason is that we add a new dimention for from-layout, but
  // that dimension is in a high probability not divisible.
  // such as [4 3 3 3] -> [4 1 3 3 4]
  int32_t TensorIndex2linear(const std::vector<int32_t> &ceil_src_shape,
                             const std::vector<int32_t> &src_shape,
                             int32_t alpha_split_pos,
                             const std::vector<int> &tensor_index) {
    auto dst_ti = tensor_index;
    if (alpha_split_pos >= 0) {
      dst_ti[alpha_split_pos] =
          dst_ti[alpha_split_pos] * ceil_src_shape[alpha_split_pos + 1] +
          dst_ti[alpha_split_pos + 1];
      for (int i = alpha_split_pos + 1; i < dst_ti.size() - 1; ++i) {
        dst_ti[i] = dst_ti[i + 1];
      }
      dst_ti.pop_back();
    }
    // alpha_pos] * alpha < non_split dim
    if (alpha_split_pos >= 0 &&
        dst_ti[alpha_split_pos] >= src_shape[alpha_split_pos]) {
      // nc4hw4, when c%4>0
      return -1;
    }
    int32_t linear_index = 0;
    size_t stride = 1;
    for (int32_t dim_index = src_shape.size() - 1; dim_index >= 0;
         --dim_index) {
      linear_index += dst_ti[dim_index] * stride;
      stride *= src_shape[dim_index];
    }
    return linear_index;
  }
  //Given a dst tensor index, to calculate src tensor index.
  std::vector<int32_t>
  transformTensorIndex(const std::vector<int32_t> &index,
                       const std::vector<int32_t> &mapping) {
    if (index.size() != mapping.size()) {
      std::cout << "trans err\n";
      return {0, 0, 0, 0};
    }
    std::vector<int32_t> dst_index = index;
    for (size_t i = 0; i < index.size(); ++i) {
      dst_index[mapping[i]] = index[i];
    }
    return dst_index;
  }
  public:
    float *DoPermute(std::string from, std::string to,
                     const std::vector<int> &src_shape, float *src){
      if (from == to)
        return src;
      auto layout_valid_checker = [](const std::string &ly) -> bool {
        size_t n = std::count_if(ly.begin(), ly.end(), [](const char &c) -> bool {
          if (!isalnum(c)) {
            return true;
          }
        });
        return n==0;
      };
      if (!layout_valid_checker(from) || !layout_valid_checker(to)) {
        return nullptr;
      }
      PermuteContext datagroup;
      bool need_reverse_permute = isdigit(from.back()) && (!isdigit(to.back()));
      datagroup.from_layout = from;
      datagroup.to_layout = to;
      datagroup.src_shape = src_shape;
      // Here, we reversed the from-layout and to-layout. in the first look,
      // it's a little bit tricky to understand. But it's nature, we just need
      // to reverse the tensor-index, it's more easier to handle the tail elements during packing
      if (need_reverse_permute) {
        swap(datagroup.from_layout, datagroup.to_layout);
        datagroup.reversed = true;
      }
      if (permute_internal(src, datagroup) < 0) {
        return nullptr;
      }
      //
      // do permute
      int dst_index = 0;
      size_t elem_size = arrayProduct(datagroup.ceil_src_shape);
      float *dst = new float[elem_size];
      while (dst_index < elem_size) {
        // dst tensorindex
        auto tensorindex = linear2TensorIndex(datagroup.dst_shape, dst_index);
        //convert to src tensor index.
        auto dst_ti = transformTensorIndex(tensorindex, datagroup.dims_to);
        int32_t src_index =
            TensorIndex2linear(datagroup.ceil_src_shape, src_shape,
                               datagroup.src_alpha_pos, dst_ti);
        // Here is the key to process reverse permute,  read every dst_index in
        // packed tensor and write back to non-packed tensor
        if (datagroup.reversed) {
          if (src_index >= 0) {
            dst[src_index] = src[dst_index];
          }
        } else {
          dst[dst_index] = src_index >= 0 ? src[src_index] : 0;
        }
        dst_index++;
      }
      return dst;
    }
};


}