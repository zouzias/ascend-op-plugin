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

at::Tensor& huber_loss_out(
        const at::Tensor& self,
        const at::Tensor& target,
        int64_t reduction,
        float delta,
        at::Tensor& output) {
    DO_COMPATIBILITY(
        aclnnHuberLoss,
        acl_op::huber_loss_out(self, target, reduction, delta, output));
    return op_api::huber_loss_forward_out(self, target, reduction, delta, output);
}

at::Tensor huber_loss(
        const at::Tensor& self,
        const at::Tensor& target,
        int64_t reduction,
        float delta) {
    return at::huber_loss_forward(self, target, reduction, delta);
}

at::Tensor& huber_loss_forward_out(
        const at::Tensor& self,
        const at::Tensor& target,
        int64_t reduction,
        float delta,
        at::Tensor& output) {
    DO_COMPATIBILITY(
        aclnnHuberLoss,
        acl_op::huber_loss_forward_out(self, target, reduction, delta, output));

    EXEC_NPU_CMD(aclnnHuberLoss, self, target, reduction, delta, output);
    return output;
}

at::Tensor huber_loss_forward(
        const at::Tensor& self,
        const at::Tensor& target,
        int64_t reduction,
        float delta) {
    DO_COMPATIBILITY(
        aclnnHuberLoss,
        acl_op::huber_loss_forward(self, target, reduction, delta));

    at::IntArrayRef output_size;
    if (reduction == at::Reduction::None) {
        output_size = self.sizes();
    } else {
        output_size = at::ArrayRef<int64_t>();
    }
    auto output = npu_preparation::apply_tensor(self, output_size);

    op_api::huber_loss_forward_out(self, target, reduction, delta, output);
    return output;
}
} // namespace op_api
