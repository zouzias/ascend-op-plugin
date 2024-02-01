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
#include "op_plugin/OpInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor mish_backward(const at::Tensor &grad_output, const at::Tensor &self)
{
    DO_COMPATIBILITY(aclnnMishBackward, acl_op::mish_backward(grad_output, self));
    auto output_size = op_infer::broadcast_ops_npu_output_size(grad_output.sizes(), self.sizes());
    at::ScalarType output_type = at::native::result_type(grad_output, self);
    at::Tensor grad_input = npu_preparation::apply_tensor_without_format(output_size,
                                                                         self.options().dtype(output_type));
    EXEC_NPU_CMD(aclnnMishBackward, grad_output, self, grad_input);
    return grad_input;
}

}  // namespace op_api

