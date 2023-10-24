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
using npu_preparation = at_npu::native::OpPreparation;

// std::tuple<at::Tensor&, at::Tensor&> rms_norm_out(
//     const at::Tensor& self,
//     const at::Tensor& gamma,
//     float eplision,
//     at::Tensor& y,
//     at::Tensor& rstd) {
//   DO_COMPATIBILITY(aclnnRmsNorm, acl_op::rms_norm_out(self, gamma, eplision, y, rstd));
//   auto output_size = op_infer::rms_norm_npu_output_size(self, gamma, eplision);
//   npu_preparation::check_tensor({self, gamma}, y, self.scalar_type(), output_size);
//   npu_preparation::check_tensor({self, gamma}, rstd, at::ScalarType::Float, output_size);

//   EXEC_NPU_CMD(aclnnRmsNorm, self, gamma, eplision, y, rstd);
//   return std::tuple<at::Tensor&, at::Tensor&>(y, rstd);
// }

std::tuple<at::Tensor, at::Tensor> rms_norm(
    const at::Tensor& self,
    const at::Tensor& gamma,
    float epsilon) {
  DO_COMPATIBILITY(aclnnRMSNorm, acl_op::rms_norm(self, gamma, epsilon));
  auto output_size = op_infer::rms_norm_npu_output_size(self, gamma, epsilon);
  at::Tensor y = npu_preparation::apply_tensor_without_format(output_size[0], self.options());
  at::Tensor rstd = npu_preparation::apply_tensor_without_format(output_size[1], self.options().dtype(at::kFloat));

  EXEC_NPU_CMD(aclnnRmsNorm, self, gamma, epsilon, y, rstd);
  return std::tuple<at::Tensor, at::Tensor>(y, rstd);
}

} // namespace op_api
