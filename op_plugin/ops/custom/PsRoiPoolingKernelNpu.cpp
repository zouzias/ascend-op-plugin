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
using npu_op_command = at_npu::native::OpCommand;

namespace {
at::Tensor& ps_roi_pooling_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t group_size,
    int64_t output_dim) {
  npu_op_command cmd;
  cmd.Name("PSROIPoolingV2")
      .Input(self, "x", ACL_FORMAT_NCHW)
      .Input(rois)
      .Output(result, "y", ACL_FORMAT_NCHW)
      .Attr("spatial_scale", (float)spatial_scale)
      .Attr("output_dim", output_dim)
      .Attr("group_size", group_size)
      .Run();

  return result;
}

at::Tensor ps_roi_pooling(
    const at::Tensor& self,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t group_size,
    int64_t output_dim) {
  auto output_size ={
      rois.size(0) * rois.size(2), output_dim, group_size, group_size};

  at::Tensor result = npu_preparation::apply_tensor(self, output_size);

  ps_roi_pooling_npu_nocheck(
      result,
      self,
      rois,
      spatial_scale,
      group_size,
      output_dim);

  return result;
}

at::Tensor& ps_roi_pooling_backward_npu_nocheck(
    at::Tensor& input_grad,
    const at::Tensor& output_grad,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t group_size,
    int64_t output_dim,
    at::IntArrayRef input_size) {
  npu_op_command cmd;
  cmd.Name("PSROIPoolingGradV2D")
      .Input(output_grad, "x", ACL_FORMAT_NCHW)
      .Input(rois)
      .Output(input_grad, "y", ACL_FORMAT_NCHW)
      .Attr("spatial_scale", (float)spatial_scale)
      .Attr("group_size", group_size)
      .Attr("output_dim", output_dim)
      .Attr("input_size", input_size)
      .Run();

  return input_grad;
}
} // namespace

at::Tensor npu_ps_roi_pooling_backward(
    const at::Tensor& output_grad,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t group_size,
    int64_t output_dim,
    c10::SymIntArrayRef size) {
  auto input_size = c10::asIntArrayRefUnchecked(size);
  auto output_size ={
      rois.size(0), group_size * group_size * output_dim, input_size[0], input_size[1]};

  at::Tensor input_grad = npu_preparation::apply_tensor(output_grad, output_size);

  ps_roi_pooling_backward_npu_nocheck(
      input_grad,
      output_grad,
      rois,
      spatial_scale,
      group_size,
      output_dim,
      input_size);

  return input_grad;
}

at::Tensor npu_ps_roi_pooling(const at::Tensor& self,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t group_size,
    int64_t output_dim) {
    return ps_roi_pooling(self, rois, spatial_scale, group_size, output_dim);
}
} // namespace op_plugin
