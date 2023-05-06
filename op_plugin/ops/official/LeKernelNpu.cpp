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

#include "op_plugin/ops/OpInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_plugin {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& le_out_npu_nocheck(const at::Tensor& self, at::Scalar other, at::Tensor& result) {
  at_npu::native::OpCommand cmd;
  cmd.Name("LessEqual")
     .Input(self)
     .Input(other, self.scalar_type())
     .Output(result)
     .Run();

  return result;
}

at::Tensor& le_out_npu_nocheck(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  auto unified_result = npu_preparation::comparison_op_check(result, self, other, true);
  at_npu::native::OpCommand cmd;
  cmd.Name("LessEqual")
     .Expect(unified_result)
     .Input(self)
     .Input(other)
     .Output(result)
     .Run();

  return result;
}
}

at::Tensor& le_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result) {
  at::Tensor formatCastOfSelf = npu_preparation::CastBackToOriFormat(self);
  auto outputSize = formatCastOfSelf.sizes();
  npu_preparation::CheckOut(
    {self},
    result,
    ACL_FORMAT_ND,
    result.scalar_type(),
    outputSize);

  le_out_npu_nocheck(formatCastOfSelf, other, result);
  return result;
}


at::Tensor& le_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  at::Tensor formatCastOfSelf = npu_preparation::CastBackToOriFormat(self);
  at::Tensor formatCastOfOther = npu_preparation::CastBackToOriFormat(other);
  auto outputSize = op_infer::broadcast_ops_npu_output_size(formatCastOfSelf, formatCastOfOther);

  npu_preparation::CheckOut(
    {self},
    result,
    ACL_FORMAT_ND,
    result.scalar_type(),
    outputSize);

  le_out_npu_nocheck(formatCastOfSelf, formatCastOfOther, result);
  return result;
}

at::Tensor le(const at::Tensor& self, const at::Scalar& other) {
  at::Tensor formatCastOfSelf = npu_preparation::CastBackToOriFormat(self);
  at::Tensor result = npu_preparation::ApplyTensorWithFormat(
    formatCastOfSelf.sizes(),
    formatCastOfSelf.options().dtype(at::kBool),
    ACL_FORMAT_ND);
  le_out_npu_nocheck(formatCastOfSelf, other, result);
  return result;
}

at::Tensor le(const at::Tensor& self, const at::Tensor& other) {
  at::Tensor formatCastOfSelf = npu_preparation::CastBackToOriFormat(self);
  at::Tensor formatCastOfOther = npu_preparation::CastBackToOriFormat(other);

  auto outputSize = op_infer::broadcast_ops_npu_output_size(formatCastOfSelf, formatCastOfOther);
  at::Tensor result = npu_preparation::ApplyTensorWithFormat(
    outputSize,
    formatCastOfSelf.options().dtype(at::kBool),
    ACL_FORMAT_ND);

  le_out_npu_nocheck(formatCastOfSelf, formatCastOfOther, result);
  return result;
}

at::Tensor& le_(at::Tensor& self, const at::Scalar& other) {
  npu_preparation::CastBackToOriFormat(self);
  npu_preparation::CheckMemory({self}, {self});
  at::Tensor result = npu_preparation::ApplyTensor(
    self,
    self.options().dtype(at::ScalarType::Byte));
  if (!npu_utils::check_match(&self)) {
    at::Tensor contiguousSelf = npu_utils::format_contiguous(self);
    le_out_npu_nocheck(contiguousSelf, other, result);
  } else {
    le_out_npu_nocheck(self, other, result);
  }
  self.copy_(result);
  return self;
}

at::Tensor& le_(at::Tensor& self, const at::Tensor& other) {
  npu_preparation::CastBackToOriFormat(self);
  at::Tensor ori_other = npu_preparation::CastBackToOriFormat(other);
  npu_preparation::CheckMemory({self, ori_other}, {self});
  at::Tensor result = npu_preparation::ApplyTensor(
    self,
    self.options().dtype(at::ScalarType::Byte));
  if (!npu_utils::check_match(&self)) {
    at::Tensor contiguousSelf = npu_utils::format_contiguous(self);
    le_out_npu_nocheck(contiguousSelf, ori_other, result);
  } else {
    le_out_npu_nocheck(self, ori_other, result);
  }
  self.copy_(result);
  return self;
}


} // namespace op_plugin
