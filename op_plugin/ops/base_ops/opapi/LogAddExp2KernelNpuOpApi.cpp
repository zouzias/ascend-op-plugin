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
#include "op_plugin/utils/KernelNpuOutputSize.h"

namespace op_api {
at::Tensor& logaddexp2_out(const at::Tensor &self, const at::Tensor &other, at::Tensor &out)
{
  DO_COMPATIBILITY(aclnnLogAddExp2, acl_op::logaddexp2_out(self, other, out));
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  at_npu::native::OpPreparation::check_tensor({self}, out, out, output_size);
  EXEC_NPU_CMD(aclnnLogAddExp2, self, other, out);

  return out;
}

at::Tensor logaddexp2(const at::Tensor &self, const at::Tensor &other)
{
  DO_COMPATIBILITY(aclnnLogAddExp2, acl_op::logaddexp2(self, other));
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  at::ScalarType output_type = at::native::result_type(self, other);
  at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(output_size,
      self.options().dtype(output_type));
  EXEC_NPU_CMD(aclnnLogAddExp2, self, other, result);

  return result;
}

} // namespace op_api

