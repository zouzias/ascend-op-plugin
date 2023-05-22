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

#include "op_plugin/ops/OpInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_plugin {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
void _conv_depthwise2d_backward_input_out_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation) {
  c10::SmallVector<int64_t, N> strides_size = {1, 1, stride[0], stride[1]};
  c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1], padding[1]};
  c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1]};
  auto input_size = self.sizes();

  at_npu::native::OpCommand cmd;
  cmd.Name("DepthwiseConv2DBackpropInput")
      .Input(input_size, at::kInt)
      .Input(weight, "filter", ACL_FORMAT_NCHW)
      .Input(grad_output, "out_backprop", ACL_FORMAT_NCHW)
      .Output(grad_input, "input_grad", ACL_FORMAT_NCHW)
      .Attr("strides", strides_size)
      .Attr("pads", paddings)
      .Attr("dilations", dilations)
      .Attr("data_format", (string)"NCHW")
      .Run();
}

void _conv_depthwise2d_backward_weight_out_nocheck(
    at::Tensor& grad_weight,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation) {
  c10::SmallVector<int64_t, N> strides_size = {1, 1, stride[0], stride[1]};
  c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1], padding[1]};
  c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1]};
  auto input_size = weight.sizes();

  at_npu::native::OpCommand cmd;
  cmd.Name("DepthwiseConv2DBackpropFilter")
      .Input(self, "input", ACL_FORMAT_NCHW)
      .Input(input_size, at::kInt)
      .Input(grad_output, "out_backprop", ACL_FORMAT_NCHW)
      .Output(grad_weight, "filter_grad", ACL_FORMAT_NCHW)
      .Attr("strides", strides_size)
      .Attr("pads", paddings)
      .Attr("dilations", dilations)
      .Attr("data_format", (string)"NCHW")
      .Run();
}
} // namespace

std::tuple<at::Tensor, at::Tensor> _conv_depthwise2d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    std::array<bool, 2> output_mask) {
  at::Tensor grad_input;
  at::Tensor grad_weight;
  if (output_mask[0]) {
    grad_input = npu_preparation::ApplyTensorWithFormat(self, ACL_FORMAT_NC1HWC0);
  }
  if (output_mask[1]) {
    grad_weight = npu_preparation::ApplyTensor(weight);
  }

  at::Tensor weight_ex = weight.permute({1, 0, 2, 3});
  if (grad_input.defined()) {
    _conv_depthwise2d_backward_input_out_nocheck(
        grad_input, grad_output, self, weight_ex, kernel_size, stride, padding, dilation);
  }
  if (grad_weight.defined()) {
    _conv_depthwise2d_backward_weight_out_nocheck(
        grad_weight, grad_output, self, weight_ex, kernel_size, stride, padding, dilation);
  }

  return std::make_tuple<at::Tensor&, at::Tensor&>(grad_input, grad_weight);
}
} // namespace op_plugin
