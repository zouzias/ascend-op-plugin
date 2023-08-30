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

at::Tensor& softplus_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& beta,
    const at::Scalar& threshold,
    at::Tensor& grad_input) {
  DO_COMPATIBILITY(aclnnSoftplusBackward, acl_op::softplus_backward_out(grad_output, self, beta, 
                                                                        threshold, grad_input));
  EXEC_NPU_CMD(aclnnSoftplusBackward, grad_output, self, beta, threshold, grad_input);
  return grad_input;
}

} // namespace op_api
