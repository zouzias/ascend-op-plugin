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

// #include "op_plugin/ops/AclOpsInterface.h"
#include "op_plugin/ops/OpApiInterface.h"
#include "op_plugin/ops/op_api/op_api_common.h"
// #include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& eq_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  // DO_COMPATIBILITY(aclnnEqTensor, NPUNativeFunctions::eq_out(self, other, result));
  auto outputSize = op_infer::broadcast_ops_npu_output_size(self, other);
  npu_preparation::check_tensor({self, other}, result, ACL_FORMAT_ND, result.scalar_type(), at::IntArrayRef(outputSize));
  EXEC_NPU_CMD(aclnnEqTensor, self, other, result);
  return result;
}

at::Tensor eq(const at::Tensor& self, const at::Tensor& other) {
  // DO_COMPATIBILITY(aclnnEqTensor, NPUNativeFunctions::eq(self, other));
  at::Tensor formatCastOfSelf = npu_preparation::CastBackToOriFormat(self);
  at::Tensor formatCastOfOther = npu_preparation::CastBackToOriFormat(other);

  // calculate the output size
  auto outputSize = op_infer::broadcast_ops_npu_output_size(formatCastOfSelf, formatCastOfOther);

  // construct the output tensor of the NPU
  at::Tensor result =
      npu_preparation::apply_tensor_without_format(outputSize, formatCastOfSelf.options().dtype(at::kBool));

  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnEqTensor, formatCastOfSelf, formatCastOfOther, result);
  return result;
}

at::Tensor& eq_out_npu_scalar(at::Tensor& result, const at::Tensor& self, at::Scalar other) {
  EXEC_NPU_CMD(aclnnEqScalar, self, other, result);
  return result;
}

at::Tensor eq(const at::Tensor& self, const at::Scalar& other) {
  // DO_COMPATIBILITY(aclnnEqScalar, NPUNativeFunctions::eq(self, other));
  at::Tensor formatCastOfSelf = npu_preparation::CastBackToOriFormat(self);

  // calculate the output size
  auto outputSize = input_same_output_size(formatCastOfSelf);

  // construct the output tensor of the NPU
  at::Tensor result =
      npu_preparation::apply_tensor_without_format(outputSize, formatCastOfSelf.options().dtype(at::kBool));

  // calculate the output result of the NPU
  eq_out_npu_scalar(result, formatCastOfSelf, other);
  return result;
}

at::Tensor& eq_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result) {
  // DO_COMPATIBILITY(aclnnEqScalar, NPUNativeFunctions::eq_out(self, other, result));
  npu_preparation::check_tensor({self}, result, ACL_FORMAT_ND, result.scalar_type(), self.sizes());

  eq_out_npu_scalar(result, self, other);

  return result;
}

at::Tensor& eq_(at::Tensor &self, const at::Tensor &other) {
  // DO_COMPATIBILITY(aclnnInplaceEqTensor, NPUNativeFunctions::eq_(self, other));

  c10::SmallVector<at::Tensor, N> inputs = {self, other};
  c10::SmallVector<at::Tensor, N> outputs = {self};
  CalcuOpUtil::CheckMemoryOverLaps(inputs, outputs);

  EXEC_NPU_CMD(aclnnInplaceEqTensor, self, other);
  return self;
}

at::Tensor& eq_(at::Tensor &self, const at::Scalar& other) {
  // DO_COMPATIBILITY(aclnnInplaceEqScalar, NPUNativeFunctions::eq_(self, other));

  EXEC_NPU_CMD(aclnnInplaceEqScalar, self, other);
  return self;
}
} // namespace op_api