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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor &binary_cross_entropy_out(
    const at::Tensor &self,
    const at::Tensor &target,
    const c10::optional<at::Tensor> &weight_opt,
    int64_t reduction,
    at::Tensor &result)
{
  DO_COMPATIBILITY(aclnnBinaryCrossEntropy,
      acl_op::binary_cross_entropy_out(self, target, weight_opt, reduction, result));
  EXEC_NPU_CMD(aclnnBinaryCrossEntropy, self, target, weight_opt, reduction, result);

  return result;
}

at::Tensor binary_cross_entropy(
    const at::Tensor &self,
    const at::Tensor &target,
    const c10::optional<at::Tensor> &weight_opt,
    int64_t reduction)
{
  DO_COMPATIBILITY(aclnnBinaryCrossEntropy,
      acl_op::binary_cross_entropy(self, target, weight_opt, reduction));
  const at::Tensor &weight = c10::value_or_else(weight_opt, [] { return at::Tensor(); });

  at::IntArrayRef outputSize;
  if (reduction == at::Reduction::None) {
    outputSize = op_infer::input_same_output_size(self);
  } else {
    outputSize = at::ArrayRef<int64_t>();
  }

  at::Tensor result = npu_preparation::apply_tensor_without_format(self, outputSize);
  EXEC_NPU_CMD(aclnnBinaryCrossEntropy, self, target, weight_opt, reduction, result);

  return result;
}

}  // namespace op_api
