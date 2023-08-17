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
using calcu_op_util = at_npu::native::CalcuOpUtil;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor binary_cross_entropy_with_logits_nocheck(
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Tensor& weight,
    const at::Tensor& pos_weight,
    int64_t reduction) {
  at::IntArrayRef output_size;
  int64_t result_format = calcu_op_util::GetTensorNpuFormat(self);

  if (reduction == at::Reduction::None) {
    output_size = self.sizes();
  } else {
    output_size = at::ArrayRef<int64_t>();
    result_format = ACL_FORMAT_ND;
  }

  at::Tensor result = npu_preparation::apply_tensor_with_format(output_size, self.options(), result_format);
  at::Tensor weight_tensor;
  if (weight.defined()) {
    weight_tensor = npu_utils::format_contiguous(weight);
    weight_tensor = (weight.scalar_type() != self.scalar_type()) ?
        op_plugin::npu_dtype_cast(weight_tensor, self.scalar_type()) : weight_tensor;
  } else {
    weight_tensor = at::ones(self.sizes(), self.options());
  }

  at::Tensor pos_weight_tensor;
  if (pos_weight.defined()) {
    pos_weight_tensor = npu_utils::format_contiguous(pos_weight);
    pos_weight_tensor = (pos_weight_tensor.scalar_type() != self.scalar_type()) ?
        op_plugin::npu_dtype_cast(pos_weight_tensor, self.scalar_type()) : pos_weight_tensor;
  } else {
    pos_weight_tensor = at::ones(self.sizes(), self.options());
  }

  std::string reduction_str = op_plugin::utils::get_reduction_str(reduction);
  at_npu::native::OpCommand cmd;
  cmd.Name("SigmoidCrossEntropyWithLogitsV2")
      .Input(self.to(target.dtype()))
      .Input(target)
      .Input(weight_tensor)
      .Input(pos_weight_tensor)
      .Output(result)
      .Attr("reduction", reduction_str)
      .Run();

  return result;
}
} // namespace

at::Tensor binary_cross_entropy_with_logits(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& pos_weight_opt,
    int64_t reduction) {
  const at::Tensor& weight = c10::value_or_else(weight_opt, [] {return at::Tensor();});
  const at::Tensor& pos_weight = c10::value_or_else(pos_weight_opt, [] {return at::Tensor();});
  return binary_cross_entropy_with_logits_nocheck(self, target, weight, pos_weight, reduction);
}
} // namespace op_plugin
