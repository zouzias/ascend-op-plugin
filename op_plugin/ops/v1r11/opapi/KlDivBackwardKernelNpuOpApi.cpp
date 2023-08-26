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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {

at::Tensor kl_div_backward(const at::Tensor& grad_output, const at::Tensor& self,
                                                    const at::Tensor& target, int64_t reduction, bool log_target) {
  DO_COMPATIBILITY(aclnnKlDivBackward,
                   acl_op::kl_div_backward(grad_output, self, target, reduction, log_target));

  at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(self);
  EXEC_NPU_CMD(aclnnKlDivBackward, grad_output, self, target, reduction, log_target, result);
  return result;
}
}  // namespace op_api
