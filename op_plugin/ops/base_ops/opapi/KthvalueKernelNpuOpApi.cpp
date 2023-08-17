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

std::tuple<at::Tensor&, at::Tensor&> kthvalue_out(
    const at::Tensor& self,
    int64_t k,
    int64_t dim,
    bool keepdim,
    at::Tensor& values,
    at::Tensor& indices) {
  DO_COMPATIBILITY(aclnnKthvalue, acl_op::kthvalue_out(self, k, dim, keepdim, values, indices));
  auto output_size = op_infer::kthvalue_npu_output_size(self, dim, keepdim);
  at_npu::native::OpPreparation::check_tensor({self}, values, self.scalar_type(), output_size);
  at_npu::native::OpPreparation::check_tensor({self}, indices, at::ScalarType::Long, output_size);

  EXEC_NPU_CMD(aclnnKthvalue, self, k, dim, keepdim, values, indices);
  return std::tuple<at::Tensor&, at::Tensor&>(values, indices);
}

std::tuple<at::Tensor&, at::Tensor&> kthvalue_out(
    const at::Tensor& self,
    int64_t k,
    at::Dimname dim,
    bool keepdim,
    at::Tensor& values,
    at::Tensor& indices) {
  DO_COMPATIBILITY(aclnnKthvalue, acl_op::kthvalue_out(self, k, dim, keepdim, values, indices));
  const int64_t real_dim = dimname_to_position(self, dim);
  auto output_size = op_infer::kthvalue_npu_output_size(self, real_dim, keepdim);
  at_npu::native::OpPreparation::check_tensor({self}, values, self.scalar_type(), output_size);
  at_npu::native::OpPreparation::check_tensor({self}, indices, at::ScalarType::Long, output_size);

  EXEC_NPU_CMD(aclnnKthvalue, self, k, real_dim, keepdim, values, indices);
  return std::tuple<at::Tensor&, at::Tensor&>(values, indices);
}

std::tuple<at::Tensor, at::Tensor> kthvalue(
    const at::Tensor& self,
    int64_t k,
    int64_t dim,
    bool keepdim) {
  DO_COMPATIBILITY(aclnnKthvalue, acl_op::kthvalue(self, k, dim, keepdim));
  auto output_size = op_infer::kthvalue_npu_output_size(self, dim, keepdim);

  at::Tensor values = at_npu::native::OpPreparation::apply_tensor_without_format(output_size, self.options());
  at::Tensor indices = at_npu::native::OpPreparation::apply_tensor_without_format(output_size, self.options().dtype(at::kLong));

  EXEC_NPU_CMD(aclnnKthvalue, self, k, dim, keepdim, values, indices);
  return std::tuple<at::Tensor, at::Tensor>(values, indices);
}

std::tuple<at::Tensor, at::Tensor> kthvalue(
    const at::Tensor& self,
    int64_t k,
    at::Dimname dim,
    bool keepdim) {
  DO_COMPATIBILITY(aclnnKthvalue, acl_op::kthvalue(self, k, dim, keepdim));
  const int64_t real_dim = dimname_to_position(self, dim);
  auto output_size = op_infer::kthvalue_npu_output_size(self, real_dim, keepdim);

  at::Tensor values = at_npu::native::OpPreparation::apply_tensor_without_format(output_size, self.options());
  at::Tensor indices = at_npu::native::OpPreparation::apply_tensor_without_format(output_size, self.options().dtype(at::kLong));

  EXEC_NPU_CMD(aclnnKthvalue, self, k, real_dim, keepdim, values, indices);
  return std::tuple<at::Tensor, at::Tensor>(values, indices);
}
} // namespace op_api
