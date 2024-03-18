// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

at::Tensor& huber_loss_backward_out(
        const at::Tensor& grad_output,
        const at::Tensor& self,
        const at::Tensor& target,
        int64_t reduction,
        float delta,
        at::Tensor& grad_input) {
    string reduction_str = op_plugin::utils::get_reduction_str(reduction);
    at_npu::native::OpCommand cmd;
    cmd.Name("HuberLossGrad")
        .Input(grad_output)
        .Input(self)
        .Input(target)
        .Output(grad_input)
        .Attr("reduction", reduction_str)
        .Attr("delta", delta)
        .Run();
    return grad_input;
}

at::Tensor& huber_loss_backward(
        const at::Tensor& grad_output,
        const at::Tensor& self,
        const at::Tensor& target,
        int64_t reduction,
        float delta) {
    auto grad_input = npu_preparation::apply_tensor(self);
    acl_op::huber_loss_backward_out(
        grad_output,
        self,
        target,
        reduction,
        delta,
        grad_input);
    return grad_input;
}
} // namespace acl_op