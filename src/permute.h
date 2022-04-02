#pragma once
#include "util.h"
#include <algorithm>
#include <iostream>
#include <map>
#include <sstream>
#include <stdint.h>
#include <string>

/*
  // what we can do now is any index permuting under fllowing conditions
  // for any packed tensor, we can handle the tailing data
  // 1. non-packed tensor -> non packed tensor, such as nchw->nhwc
  // 2. non-packed tensor -> packed tensor, such as nchw->nhc4w4
  // 3. packed tensor -> packed tensor, such as nc4hw4->nhc4w4, only the
  // identical packed-index is spported
  // 4. packed tensor -> non-packed tensor, such as nc4hw4 -> nchw

the same time, in order to support GPU code generator, we need to distingguish
what the src_mem or dst_mem is. we have to handle that with different ways. Like
buffer/image2d, the first one is linear, we can treat it as a normal memory,
while image2d is uncontinous, we have to know which dimentions stored in the
image-width, and which dimentions in image-height.

/-------------------------------------------/ buffer

/++++++++++++
+++++++++++++
+++++++++++++
+++++++++++++
+++++++++++++/ image

we will use a delimeter '|' to tell how to map a 4-dim tensor to a 2-dim image plane

*/
namespace Tensor {

/*
a contenxt data structure to store intermediate infos
*/
class PermuteContext {
public:
  std::vector<int> ceil_src_shape; // the intermediate layout, get from src_shape
  std::vector<int> src_shape; //src_tensor shape
  std::vector<int> dst_shape;
  std::vector<int> dims_to;// how src index map to dst index
  std::vector<int> dims_from; //
  int32_t src_alpha_pos; //the packed index position
  int32_t dst_alpha_pos;// same 
  std::string from_layout; //
  std::string to_layout;
  int32_t img_w_from_dim = -1;// for texture memory type, especially for image2d. unused in CPUpermute
  bool reversed = false; // if we need  to reverse the transpose linear index
};

enum class LayoutPackMode {
  None, // non of src or dst tensor is packed
  From, //from packed tensor to non-packed tensor
  To, //from non-packed tensor to packed tensor
  Both,// both packed
};

class PermuteBase {
public:
  //a interface for concrete class to do permute
  //virtual int32_t DoPermute(std::string from, std::string to,
  //                       const std::vector<int> &src_shape, float *src) = 0;
  // packed tensor -> packed tensor
  // for example, nc4hw4->nhc4w4
  int32_t permute_for_both_packed(float *src, PermuteContext &datag) {
    std::string &from = datag.from_layout;
    std::string &to = datag.to_layout;
    size_t f_pos =
        std::find_if(from.begin(), from.end(), isdigit) - from.begin();
    size_t t_pos = std::find_if(to.begin(), to.end(), isdigit) - to.begin();
    assert(f_pos & t_pos);
    if (from[f_pos - 1] != to[t_pos - 1]) {
      std::cout << " can't serve such permute: from " << from << "-> to " << to
                << "\n";
      return -1;
    }
    std::vector<int> &src_shape = datag.src_shape;
    from.erase(from.begin() + f_pos);
    to.erase(to.begin() + t_pos);
    return 0;
  }


  bool image2d_or_pack_permute_check(const std::string &from, const std::string &to) const {
    //'|' is the image2d dimention delimeter
    auto fp = from.find('|');
    size_t from_del_num = std::count(from.begin(), from.end(), '|');
    size_t to_del_num = std::count(to.begin(), to.end(), '|');
    //if we find '1' in both from and to, it's illegal
    if (from_del_num > 0 && to_del_num > 0) {
      std::cout << "not support image 2 image\n";
      return false;
    }
    if (from_del_num > 1) {
      std::cout << "illegal image2d dim slit" << from << "\n";
      return false;
    }
    if (to_del_num > 1) {
      std::cout << "illegal image2d dim slit" << to << "\n";
      return false;
    }
    // nhc4w4, two packing number is essential, and the two must be equal
    auto pack_piece_check = [](const std::string layout) -> bool {
      std::vector<char> digits;
      std::for_each(layout.begin(), layout.end(), [&digits](const char c) {
        if (isdigit(c)) {
          digits.push_back(c);
        }
      });
      if (digits.size() != 2 || digits[0] != digits[1]) {
        return false;
      }
      return true;
    };

    if (pack_piece_check(from) == false || pack_piece_check(to) == false) {
      std::cout << __LINE__ << " error permute: " << from << "->" << to << "\n";
      return false;
    }
    return true;
  }

  LayoutPackMode tensor_pack_mode_probe(const std::string &from,
                                        const std::string &to) const {
    LayoutPackMode pack_mode = LayoutPackMode::None;
    if (isdigit(from.back())) {
      if (isdigit(to.back())) {
        pack_mode = LayoutPackMode::Both;
      } else {
        pack_mode = LayoutPackMode::From;
      }
    } else if (isdigit(to.back())) {
      pack_mode = LayoutPackMode::To;
    }
    return pack_mode;
  }

  // because our implementation is based on permuted, which means all tensors
  // must have the same dimention. while the pack_mode will change one of
  // tensors' shape.  for example nchw->nc4hw4, src_shape is 4-dim tensor, but
  // dst_shape is 5-dims tensor.
  // hence, our solution is to process the input tensor and add it a new
  // dimention to 5-dim tensor, so we can perform permute operation happy.
  // before                 |..........nchw->nc4hw4............|
  // then                   |..........nc44hw->nc4hw4..........|
  // after normallization   |..........nc4hw->nchw4............|
  //
  int32_t normallize_layout_pack_representation(PermuteContext &datag,
                                                const LayoutPackMode& pack_mode,
                                                std::string &from,
                                                std::string &to) {
    std::string& pack_ly_ref = to;
    std::string &non_pack_ly_ref = from;
    if (pack_mode == LayoutPackMode::From) {
      pack_ly_ref = from;
      non_pack_ly_ref = to;
    }
    // if tensor is not packed, just return
    if (!isdigit(pack_ly_ref.back())) {
      return 0;
    }
    // align layout representation, we always make layout in the same dimention
    // to: nc4hw4->nchw4
    // from: nchw -> nc4hw
    int32_t alpha = 1; // how much the small pack piece is, 
    std::vector<size_t> digit_pos;
    //we have checked that there must be two digits.
    for (size_t i = 0; i < pack_ly_ref.size(); ++i) {
      if (isdigit(pack_ly_ref[i])) {
        digit_pos.push_back(i);
      }
    }

    if (pack_mode == LayoutPackMode::To) {
      datag.dst_alpha_pos = static_cast<int32_t>(digit_pos[0]) - 1;
    } else {
      datag.src_alpha_pos = static_cast<int32_t>(digit_pos[0]) - 1;
    }
    alpha = pack_ly_ref[digit_pos[0]] - '0';
    //we remove the first digit, as we already know this index was packed
    pack_ly_ref.erase(pack_ly_ref.begin() + digit_pos[0]);

    for (int i = 0; i < non_pack_ly_ref.size(); ++i) {
      //find where is the correspond axis from another layout
      if (non_pack_ly_ref[i] == pack_ly_ref[digit_pos[0] - 1]) {
        //we make it as the same packing layout, but it's continious
        non_pack_ly_ref.insert((non_pack_ly_ref).begin() + i + 1, (pack_ly_ref).back());
        if (pack_mode == LayoutPackMode::To) {
          //change the shape as well
          datag.ceil_src_shape.insert(datag.ceil_src_shape.begin() + i + 1,
                                      alpha);
          //handle the remaining elements
          datag.ceil_src_shape[i] = CeilDiv(datag.ceil_src_shape[i], alpha);
          datag.src_alpha_pos = i;
        } else {
          datag.dst_alpha_pos = i;
        }
        break;
      }
    }
    return 0;
  }

  int32_t permute_internal(float *src, PermuteContext &datag) {
    // handle nc4hw4
    const std::vector<int> &src_shape = datag.src_shape;
    std::string &from = datag.from_layout;
    std::string &to = datag.to_layout;
    if (image2d_or_pack_permute_check(from, to) == false) {
      return -1;
    }
    LayoutPackMode pack_mode = tensor_pack_mode_probe(from, to);

    // canonicalize to upper case
    std::transform(from.begin(), from.end(), from.begin(), std::toupper);
    std::transform(to.begin(), to.end(), to.begin(), std::toupper);

    datag.src_alpha_pos = -1;
    datag.dst_alpha_pos = -1;
    datag.ceil_src_shape = src_shape;
    // handle image width_dim delimeter |
    std::string *pack_ly_ref = &to;
    std::string *non_pack_ly_ref = &from;
    if (pack_mode == LayoutPackMode::From) {
      pack_ly_ref = &from;
      non_pack_ly_ref = &to;
    }
    if (pack_mode != LayoutPackMode::Both) {
      // handle image delimeter first
      auto image2d_split_pos = pack_ly_ref->find('|');
      if (image2d_split_pos != pack_ly_ref->npos) {
        auto it = pack_ly_ref->erase(pack_ly_ref->begin() + image2d_split_pos);
        datag.img_w_from_dim = (*it);
      }
      
      if (src_shape.size() == from.size() ||
          (isdigit((from).back()) && src_shape.size() + 1 == from.size())) {
      } else {
        assert(false);
      }
      if (0 !=
          normallize_layout_pack_representation(datag, pack_mode, from, to)) {
        return -1;
      }
    } else if (pack_mode == LayoutPackMode::Both) {
      assert(permute_for_both_packed(src, datag) == 0);
    }
    // the last validation, they must have the same axis
    {
      auto f = from, t = to;
      sort(f.begin(), f.end());
      sort(t.begin(), t.end());
      if (f != t) {
        std::cout << __LINE__ << "error permute " << from << "->" << to << "\n";
        return -1;
      }
    }
    // it's a little weried, we have to locate the real image2d index after all
    // layout was normallized
    // we just recored its axis charactor above,
    for (int32_t i = 0; i < (*pack_ly_ref).size() && datag.img_w_from_dim != -1;
         ++i) {
      // if datag.img_w_from_dim==-1. it's fine.
      if ((*pack_ly_ref)[i] == datag.img_w_from_dim) {
        datag.img_w_from_dim = i;
        break;
      }
    }
    std::map<char, int> c2dim;
    for (int i = 0; i < from.size(); ++i) {
      c2dim[from[i]] = i;
    }

    for (int i = 0; i < to.size(); ++i) {
      datag.dims_to.push_back(c2dim[to[i]]);
      datag.dst_shape.push_back(datag.ceil_src_shape[datag.dims_to.back()]);
    }
    return 0;
  }
};

}