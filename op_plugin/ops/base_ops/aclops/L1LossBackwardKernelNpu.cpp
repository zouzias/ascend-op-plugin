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

at::Tensor l1_loss_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {
  at::Tensor result = npu_preparation::apply_tensor(self);
  at::Tensor grad_output_broadcast =
      grad_output.sizes() != self.sizes() ? op_plugin::npu_broadcast(grad_output, self.sizes()) : grad_output;
  at::Tensor target_broadcast =
      target.sizes() != self.sizes() ? op_plugin::npu_broadcast(target, self.sizes()) : target;

  std::string reduction_str = calcu_op_util::Getreduction_str(reduction);
  at_npu::native::OpCommand cmd;
  cmd.Name("L1LossGrad")
      .Input(grad_output_broadcast)
      .Input(self)
      .Input(target_broadcast)
      .Attr("reduction", reduction_str)
      .Output(result)
      .Run();
  return result;
}
} // namespace op_plugin
