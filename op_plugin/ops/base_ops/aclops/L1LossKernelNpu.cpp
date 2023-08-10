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

namespace {
at::Tensor& l1_loss_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {
  std::string reduction_str = calcu_op_util::GetReductionStr(reduction);
  at_npu::native::OpCommand cmd;
  cmd.Name("LpLoss")
      .Input(self)
      .Input(target)
      .Attr("reduction", reduction_str)
      .Attr("p", (int64_t)1)
      .Output(result)
      .Run();
  return result;
}
} // namespace

at::Tensor npu_l1_loss(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {
  at::IntArrayRef output_size;
  if (reduction == at::Reduction::None) {
    output_size = op_infer::input_same_output_size(self);
  }
  at::Tensor result = npu_preparation::ApplyTensor(self, output_size);
  l1_loss_nocheck(result, self, target, reduction);
  return result;
}

at::Tensor l1_loss(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {
  return npu_l1_loss(self, target, reduction);
}
} // namespace op_plugin
