// Copyright (c) 2023, Huawei Technologies.All rights reserved.
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

at::Tensor& huber_loss_backward_out(
        const at::Tensor& grad_output,
        const at::Tensor& self,
        const at::Tensor& target,
        int64_t reduction,
        float delta,
        at::Tensor& grad_input) {
    DO_COMPATIBILITY(
        aclnnHuberLossBackward,
        acl_op::huber_loss_backward_out(grad_output, self, target, reduction, delta, grad_input));

    EXEC_NPU_CMD(aclnnHuberLossBackward, grad_output, self, target, reduction, delta, grad_input);
    return grad_input;
}

at::Tensor huber_loss_backward(
        const at::Tensor& grad_output,
        const at::Tensor& self,
        const at::Tensor& target,
        int64_t reduction,
        float delta) {
    DO_COMPATIBILITY(
        aclnnHuberLossBackward,
        acl_op::huber_loss_backward(grad_output, self, target, reduction, delta));

    auto grad_input = npu_preparation::apply_tensor(self);
    op_api::huber_loss_backward_out(grad_output, self, target, reduction, delta, grad_input);
    return grad_input;
}
} // namespace op_api
