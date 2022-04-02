#include "permute.h"

namespace Tensor {

//Image2d has two attributes
struct ImageAttribute {
  int32_t width;
  int32_t height;
};
class OpenClCode {
public:
  std::string source_code;
  ImageAttribute attr;
  std::string kernel_name;
};

/*
OPenCL tensor permute is a little bit different than CPU, since the different
memory type.
for buffer->buffer, the same with CPU
for buffer->image2d or image2d-buffer, it's special to read/write data
image2d->image2d is not support, becuase, image2d requires the minimal data-width is 4 elements

*/
class PermuteOpenCL :public PermuteBase{
public:
  class MemoryType {
  public:
    bool Image = false;
    int32_t width_from_dim_ = 0;
  };
  class BufferMemory : public MemoryType {
  public:
    BufferMemory() { Image = false; }
  };
  class ImageMemory : public MemoryType {
  public:
    ImageMemory(int n_dim) {
      if (n_dim <= 0) {
        throw;
      }
      Image = true;
      width_from_dim_ = n_dim;
    }
  };
  OpenClCode DoPermute(std::string from, std::string to,
                       const std::vector<int> &src_shape,
                       float *src) {
    OpenClCode clartifacts;
    //we use '|' to represent memory location of tensor is Image2D or not
    auto fp = from.find('|');
    auto tp = to.find('|');
    if (fp != from.npos && tp != to.npos) {
      std::cout << "not support image 2 image\n";
      return clartifacts;
    }
    PermuteContext datagroup;
    datagroup.from_layout = from;
    datagroup.to_layout = to;
    datagroup.src_shape = src_shape;
    if (fp != from.npos && tp == to.npos) {
      swap(datagroup.from_layout, datagroup.to_layout);
      datagroup.reversed = true;
    }
    if (permute_internal(src, datagroup) != 0) {
      return clartifacts;
    }
    //we have to handle both buffer and image2d memory
    MemoryType intype, outtype;
    if (fp != from.npos) {
      intype = ImageMemory(datagroup.img_w_from_dim);
    } else {
      intype = BufferMemory();
    }

    if (tp != to.npos) {
      // because we reversed from/to
      outtype = ImageMemory(datagroup.img_w_from_dim);
    } else {
      outtype = BufferMemory();
    }
    clartifacts = layout_transform_codegen_opencl(datagroup, intype, outtype);
    return clartifacts;
  }

private:
  //
  std::string
  generate_image_index_tensorindex(const std::vector<int32_t> &shape_width,
                                   const std::vector<int32_t> &mapping,
                                   std::string base_var, std::string dimmap,
                                   std::vector<std::string> &var_load,
                                   int32_t split_pos) {
    std::string var = "dim_";
    std::string cur = "";
    std::string space_head = "    ";
    int32_t ind = var_load.size();
    std::ostringstream oss;
    int32_t last_index = shape_width.size() - 1;
    for (int i = last_index; i >= 0; i--, ind++) {
      if (i == last_index) {
        cur = base_var;
      }
      var_load.push_back(
          var + std::string(1, dimmap[mapping[mapping.size() - 1 - ind]]));
      std::string dim_times = "";
      if (mapping.size() - 1 - ind == split_pos) {
        dim_times = "* 4";
      }
      if (i == 0) {
        oss << space_head << "const int " << var_load.back() << " = " << cur
            << dim_times << "; \n";
        continue;
      } else {
        oss << space_head << "const int " << var_load.back() << " = (" << cur
            << " % " << shape_width[i] << ")" << dim_times << "; \n";
        cur = "(" + cur + "/" + std::to_string(shape_width[i]) + ")";
      }
    }
    oss << "\n";
    return oss.str();
  }

  OpenClCode layout_transform_codegen_opencl(PermuteContext &datagroup,
                                             MemoryType intype,
                                             MemoryType outtype) {
    OpenClCode out_artifacts;
    const std::vector<int> &src_shape = datagroup.src_shape;
    const std::vector<int> &dst_shape = datagroup.dst_shape;
    const std::vector<int32_t> &mapping = datagroup.dims_to;
    int32_t dst_alpha_pos = datagroup.dst_alpha_pos;
    assert(dst_shape.back() == 4);
    // image to image is not surpported
    // buffer to buffre ok
    // buffer to image  ok
    // image to buffer  ok
    assert(!(intype.Image && outtype.Image));
    if (intype.Image) {
      assert(datagroup.reversed);
    } else if (outtype.Image) {
      assert(!datagroup.reversed);
    }
    std::ostringstream kernel_oss;
    kernel_oss
        << R"dec(__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
#define SELECT_PREDICATE int // this is for select predicate cast
#define FLOAT float
#define FLOAT4 float4
#define CONVERT_FLOAT convert_float
#define CONVERT_FLOAT4 convert_float4
#define RI_F(image, coord) read_imagef((image), (SAMPLER), (coord))
#define WI_F(image, coord, value) write_imagef((image), (coord), (value))
)dec";
    const std::string store_vec4_to_buffer = R"dec(
    // Safely scatter store a 4-element vector to global memory
#define SAFE_SCATTER_STG_VEC4(output, base_offset, stride, remain, v) \
  {                                                                   \
    int r = (remain);                                                 \
    if (r > 0) {                                                      \
      int i = base_offset;                                            \
      if (r >= 4) {                                                   \
        (output)[i] = (v).s0;                                         \
        i += stride;                                                      \
        (output)[i] = (v).s1;                                         \
        i += stride;                                                      \
        (output)[i] = (v).s2;                                         \
        i += stride;                                                      \
        (output)[i] = (v).s3;                                         \
      } else if (r == 3) {                                            \
        (output)[i] = (v).s0;                                         \
        i += stride;                                                      \
        (output)[i] = (v).s1;                                         \
        i += stride;                                                      \
        (output)[i] = (v).s2;                                         \
      } else if (r == 2) {                                            \
        (output)[i] = (v).s0;                                         \
        i += stride;                                                      \
        (output)[i] = (v).s1;                                         \
      } else if (r == 1) {                                            \
        (output)[i] = (v).s0;                                         \
      }                                                               \
    }                                                                 \
  }
)dec";
    const std::string load_vec4_from_buffer = R"dec(
// Safely gather load a 4-element vector from global memory
#define SAFE_GATHER_LDG_VEC4(v, input, base_offset, stride, remain) \
  {                                                                 \
    int r = (remain);                                               \
    if (r > 0) {                                                    \
      int i = (base_offset);                                        \
      if (r >= 4) {                                                 \
        (v).s0 = (input)[i];                                        \
        i += (stride);                                              \
        (v).s1 = (input)[i];                                        \
        i += (stride);                                              \
        (v).s2 = (input)[i];                                        \
        i += (stride);                                              \
        (v).s3 = (input)[i];                                        \
      } else if (r == 3) {                                          \
        (v).s0 = (input)[i];                                        \
        i += (stride);                                              \
        (v).s1 = (input)[i];                                        \
        i += (stride);                                              \
        (v).s2 = (input)[i];                                        \
      } else if (r == 2) {                                          \
        (v).s0 = (input)[i];                                        \
        i += (stride);                                              \
        (v).s1 = (input)[i];                                        \
      } else if (r == 1) {                                          \
        (v).s0 = (input)[i];                                        \
      }                                                             \
    }                                                               \
  }
)dec";
    // load macro definition
    if (outtype.Image) {
      kernel_oss << load_vec4_from_buffer;
    } else if (intype.Image) {
      kernel_oss << store_vec4_to_buffer;
    }
    // produce opencl kernel name
    out_artifacts.kernel_name = "Copy";
    if (intype.Image) {
      out_artifacts.kernel_name += "Image";
    } else {
      out_artifacts.kernel_name += "Buffer";
    }
    out_artifacts.kernel_name +=
        datagroup.reversed ? datagroup.to_layout : datagroup.from_layout + "To";
    if (outtype.Image) {
      out_artifacts.kernel_name += "Image";
    } else {
      out_artifacts.kernel_name += "Buffer";
    }
    out_artifacts.kernel_name +=
        datagroup.reversed ? datagroup.from_layout : datagroup.to_layout;
    // generate kernel function signature
    kernel_oss << "__kernel void " << out_artifacts.kernel_name << "(";
    if (intype.Image) {
      kernel_oss << "__read_only image2d_t data, ";
    } else {
      kernel_oss << "__global const float* data, ";
    }
    if (outtype.Image) {
      kernel_oss << "__write_only image2d_t output){\n";
    } else {
      kernel_oss << "__global float* output){\n";
    }
    // image2d supported only.
    if (outtype.width_from_dim_ == -1 && outtype.Image) {
      return out_artifacts;
    }
    if (intype.width_from_dim_ == -1 && intype.Image) {
      return out_artifacts;
    }
    int32_t space_count = 4;
    std::string space_head = std::string(space_count, ' ');
    kernel_oss << space_head << "int x = get_global_id(0);\n";
    kernel_oss << space_head << "int y = get_global_id(1);\n";
    // image memory width whenever in/out
    out_artifacts.attr.width = 1;
    std::vector<int32_t> shape_width;
    std::vector<int32_t> shape_height;
    int32_t width_start_dim = -1;
    if (intype.Image) {
      width_start_dim = intype.width_from_dim_;
    } else if (outtype.Image) {
      width_start_dim = outtype.width_from_dim_;
    }
    // remember, we reversed in/out if input is image type memory.
    // so datagroup alwasy assume output-mem is image or both in/out are buffer
    const std::vector<int> *ref_pack_shape = nullptr;
    int32_t alpha_pos = -1;
    std::string tensor_infer_layout = datagroup.from_layout;
    // if output-mem is not image-type, then both input/output are buffer
    if (outtype.Image || intype.Image) {
      ref_pack_shape = &dst_shape;
      alpha_pos = datagroup.dst_alpha_pos;
    }
    // image sub-dimention is rgba/vec4
    int rgba_pack4 = (alpha_pos >= 0);

    if (ref_pack_shape) {
      for (int32_t i = width_start_dim;
           i < (*ref_pack_shape).size() - rgba_pack4; i++) {
        out_artifacts.attr.width *= (*ref_pack_shape)[i];
        shape_width.push_back((*ref_pack_shape)[i]);
      }
      out_artifacts.attr.height = 1;

      for (int32_t i = 0; i < width_start_dim; i++) {
        out_artifacts.attr.height *= (*ref_pack_shape)[i];
        shape_height.push_back((*ref_pack_shape)[i]);
      }
    } else {
      shape_width = dst_shape;
    }
    // exclude unvalid read/write
    kernel_oss << space_head << "if (x >= " << out_artifacts.attr.width
               << "|| y >= " << out_artifacts.attr.height << ") {return;}\n";
    int32_t ind = 0;
    std::vector<std::string> var_load;
    if (rgba_pack4) {
      var_load.push_back("0");
    }
    // when image memory involved,mapping is inrelavant of in/out, it only cares
    // image/buffer, image maps to buffer
    kernel_oss << generate_image_index_tensorindex(
        shape_width, mapping, "x", tensor_infer_layout, var_load, alpha_pos);
    kernel_oss << generate_image_index_tensorindex(
        shape_height, mapping, "y", tensor_infer_layout, var_load, alpha_pos);
    if (rgba_pack4) {
      var_load.erase(var_load.begin());
    }
    char sort_key[256] = {127};
    int ind_s = 0;
    for (auto c : datagroup.from_layout) {
      sort_key[c] = ind_s++;
    }
    sort(var_load.begin(), var_load.end(),
         [&sort_key](const std::string &a, const std::string &b) {
           return sort_key[a.back()] < sort_key[b.back()];
         });
    // H * W
    std::ostringstream oss;
    assert(var_load.size() == src_shape.size());
    auto src_stride = getStride(src_shape);
    for (int32_t i = 0; i < src_stride.size(); ++i) {
      oss << var_load[i] << "*" << src_stride[i];
      if (i != src_stride.size() - 1) {
        oss << "+";
      }
    }
    kernel_oss << space_head
               << "const int stride = " << src_stride[datagroup.src_alpha_pos]
               << ";\n"
               << space_head << "const int base_index = (" << oss.str()
               << ");\n"; // C* n + c)* HW + W * h + w; ";
    // kernel_oss<< space_head<< "printf(\"%d,%d,%d   \", x, y, base_index);\n";
    oss.str("");
    // outtype image, assuming vec_width_out= 4;
    if (outtype.Image) {
      kernel_oss << space_head
                 << "float4 v = 0;  // NOTE: buffer r/w always assume fp32\n";
      // TODO; only channel splilt is surpported
      kernel_oss << space_head
                 << "SAFE_GATHER_LDG_VEC4(v, data, base_index, stride, "
                 << src_shape[datagroup.src_alpha_pos] << "-"
                 << var_load[datagroup.src_alpha_pos] << "); \n";
      // kernel_oss << "printf(\"%.0f,%.0f,%.0f,%.0f   \",v.x,v.y,v.z,v.w);\n";
      kernel_oss << space_head
                 << "WI_F(output, (int2)(x, y), CONVERT_FLOAT4(v));\n";
    } else if (intype.Image) {
      kernel_oss
          << space_head
          << "const float4 v = convert_float4(RI_F(data, (int2)(x, y)));\n";
      kernel_oss << space_head
                 << "SAFE_SCATTER_STG_VEC4(output, base_index, stride,"
                 << src_shape[datagroup.src_alpha_pos] << "-"
                 << var_load[datagroup.src_alpha_pos] << ", v);\n";
    } else {
    }
    kernel_oss << "}\n";
    out_artifacts.source_code = kernel_oss.str();
    return out_artifacts;
  }
};
}