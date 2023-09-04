// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2013, Facebook CORPORATION.
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

at::Tensor& sinc_out(const at::Tensor& self, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnSinc, acl_op::sin_out(self, result));
  npu_preparation::check_tensor({self}, result, result.scalar_type(), self.sizes());
  EXEC_NPU_CMD(aclnnSinc, self, result);
  return result;
}

at::Tensor& sinc_(at::Tensor& self) {
  DO_COMPATIBILITY(aclnnInplaceSinc, acl_op::sin_(self));
  EXEC_NPU_CMD(aclnnInplaceSinc, self);
  return self;
}

at::Tensor sinc(const at::Tensor& self) {
  DO_COMPATIBILITY(aclnnSinc, acl_op::sin(self));
  auto output_size = self.sizes();
  auto out_dtype = self.dtype();
  if (isIntegralType(self.scalar_type(), true)) {
    out_dtype = at::kFloat;
  }
  at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(out_dtype));
  EXEC_NPU_CMD(aclnnSinc, self, result);
  return result;
}

} // namespace at_npu

