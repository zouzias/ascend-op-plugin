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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

at::Tensor& addmm_out(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& result) {
  at::Tensor mul_result = at::mul(mat1, alpha);
  at::Tensor mm_result = at::mm(mul_result, mat2);

  // matmul*alpha+self*beta
  at::add_out(result, mm_result, self, beta);
  return result;
}

at::Tensor addmm(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  auto output_size = op_infer::addmm_npu_output_size(self, mat1, mat2, beta, alpha);

  // add supports NZ with 1 dimension, and this axis can be added by ND divisible by 16,
  // then directly get NZ result
  int64_t res_format = (self.dim() == 1 && self.size(0) % 16 == 0 && self.scalar_type() == at::kHalf) ?
     ACL_FORMAT_FRACTAL_NZ : ACL_FORMAT_ND;
  at::Tensor result = npu_preparation::ApplyTensorWithFormat(output_size, self.options(), res_format);

  acl_op::addmm_out(self, mat1, mat2, beta, alpha, result);
  return result;
}

at::Tensor& addmm_(
    at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  npu_preparation::CheckMemory({self, mat1, mat2}, {self});
  if (!npu_utils::check_match(&self)) {
    at::Tensor contiguous_self = npu_utils::format_contiguous(self);
    acl_op::addmm_out(contiguous_self, mat1, mat2, beta, alpha, contiguous_self);
    npu_utils::format_fresh_view(self, contiguous_self);
  } else {
    acl_op::addmm_out(self, mat1, mat2, beta, alpha, self);
  }
  return self;
}
} // namespace acl_op
