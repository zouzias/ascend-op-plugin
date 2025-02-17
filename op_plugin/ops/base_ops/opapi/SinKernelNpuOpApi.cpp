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

at::Tensor& sin_out(const at::Tensor& self, at::Tensor& result)
{
    DO_COMPATIBILITY(aclnnSin, acl_op::sin_out(self, result));
    TORCH_CHECK(!isIntegralType(result.scalar_type(), true),
                "result dtype can't be cast to the desired output type.\n", OPS_ERROR(ErrCode::TYPE));
    auto outputSize = self.sizes();
    npu_preparation::check_tensor({self}, result, result.scalar_type(), outputSize);
    EXEC_NPU_CMD(aclnnSin, self, result);
    return result;
}

at::Tensor &sin_(at::Tensor &self)
{
    DO_COMPATIBILITY(aclnnInplaceSin, acl_op::sin_(self));
    TORCH_CHECK(!isIntegralType(self.scalar_type(), true),
                "result dtype can't be cast to the desired output type.\n", OPS_ERROR(ErrCode::TYPE));
    EXEC_NPU_CMD(aclnnInplaceSin, self);
    return self;
}

at::Tensor sin(const at::Tensor& self)
{
  auto outputSize = self.sizes();
  auto outDtype = self.dtype();
  DO_COMPATIBILITY(aclnnSin, acl_op::sin(self));
  if (isIntegralType(self.scalar_type(), true)) {
    outDtype = at::kFloat;
  }
  at::Tensor result = npu_preparation::apply_tensor_without_format(outputSize, self.options().dtype(outDtype));
  EXEC_NPU_CMD(aclnnSin, self, result);
  return result;
}

} // namespace at_npu

