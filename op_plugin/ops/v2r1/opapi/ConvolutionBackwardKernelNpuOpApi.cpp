// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

static inline c10::SmallVector<int64_t, op_infer::N> expand_dim_if_needed(
    at::IntArrayRef list_param,
    const char* param_name,
    int64_t expected_dim)
{
  if (list_param.size() == 1) {
    c10::SmallVector<int64_t, op_infer::N> expand_dim_param_vec;
    for (int64_t i = 0; i < expected_dim; i++) {
      expand_dim_param_vec.emplace_back(list_param[0]);
    }
    return expand_dim_param_vec;
  } else {
    return op_plugin::utils::convert_array_to_vector(list_param);
  }
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor> _calc_convolution_backward(
    const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight,
    const at::OptionalIntArrayRef bias_sizes_opt, at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups,
    ::std::array<bool, 3> output_mask)
{
  int64_t k = weight.ndimension();
  int64_t dim = k - 2;
  int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowConvHF32());

  // CheckForbidInternalFormat = False: turn on private format；CheckJitDisable = False: turn on JitCompile
  if ((!at_npu::native::env::CheckForbidInternalFormat() || !at_npu::native::env::CheckJitDisable())) {
    return acl_op::convolution_backward(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation,
                                        transposed, output_padding, groups, output_mask);
  }

  c10::SmallVector<int64_t, op_infer::N> stride_expand = expand_dim_if_needed(stride, "stride", dim);
  stride = at::IntArrayRef(stride_expand);

  c10::SmallVector<int64_t, op_infer::N> padding_expand = expand_dim_if_needed(padding, "padding", dim);
  padding = at::IntArrayRef(padding_expand);

  c10::SmallVector<int64_t, op_infer::N> dilation_expand = expand_dim_if_needed(dilation, "dilation", dim);
  dilation = at::IntArrayRef(dilation_expand);

  c10::SmallVector<int64_t, op_infer::N> output_padding_expand = expand_dim_if_needed(output_padding, "output_padding",
                                                                                      dim);
  output_padding = at::IntArrayRef(output_padding_expand);

  auto outputSizes = op_infer::conv2d_backward_npu_output_size(input, grad_output, weight, stride, padding, dilation,
                                                               groups);

  // construct the output tensor of the NPU
  at::Tensor gradInput;
  at::Tensor gradWeight;
  at::Tensor gradBias;

  gradInput = npu_preparation::apply_tensor_without_format(std::get<0>(outputSizes), input.options());
  gradWeight = npu_preparation::apply_tensor_without_format(std::get<1>(outputSizes), weight.options());

  // use 2nd dimension of outputSizes
  gradBias = npu_preparation::apply_tensor_without_format(std::get<2>(outputSizes), grad_output.options());

  int64_t input_dim = input.ndimension();
  at::optional<c10::IntArrayRef> bias_sizes = c10::nullopt;
  if (bias_sizes_opt.has_value()) {
    bias_sizes = bias_sizes_opt.value();
  }
  EXEC_NPU_CMD(aclnnConvolutionBackward, grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed,
               output_padding, groups, output_mask, cube_math_type, gradInput, gradWeight, gradBias);
  return std::make_tuple(std::move(gradInput), std::move(gradWeight), std::move(gradBias));
}

// length of output_mask is 3
std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::OptionalIntArrayRef bias_sizes_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups,
    std::array<bool, 3> output_mask) {
  DO_COMPATIBILITY(aclnnConvolutionBackward, acl_op::convolution_backward(grad_output, input, weight, bias_sizes_opt,
                                                                          stride, padding, dilation, transposed,
                                                                          output_padding, groups, output_mask));
  return _calc_convolution_backward(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation,
                                    transposed, output_padding, groups, output_mask);
}

} // namespace op_api
