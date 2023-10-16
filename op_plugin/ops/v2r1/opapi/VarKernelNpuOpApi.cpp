// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
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
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& var_out(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool unbiased,
    bool keepdim,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnVar, acl_op::var_out(self, dim, unbiased, keepdim, result));
  c10::SmallVector<int64_t, op_infer::SIZE> real_dim = {};
  if (dim.has_value()) {
    real_dim = op_infer::array_to_small_vector(dim.value());
  }
  auto output_size = op_infer::reduce_ops_npu_output_size(self, real_dim, keepdim);

  at_npu::native::OpPreparation::check_tensor(
      {self},
      result,
      self,
      output_size);

  auto rd = at::IntArrayRef(real_dim);
  EXEC_NPU_CMD(aclnnVar, self, rd, unbiased, keepdim, result);
  return result;
}

at::Tensor& var_out(
    const at::Tensor& self,
    at::DimnameList dim,
    bool unbiased,
    bool keepdim,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnVar, acl_op::var_out(self, dim, unbiased, keepdim, result));
  return op_api::var_out(self, dimnames_to_positions(self, dim), unbiased, keepdim, result);
}

at::Tensor& var_out(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<c10::Scalar>& correction,
    bool keepdim,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnVarCorrection, acl_op::var_out(self, dim, correction, keepdim, result));
  c10::SmallVector<int64_t, op_infer::SIZE> real_dim = {};
  if (dim.has_value()) {
    real_dim = op_infer::array_to_small_vector(dim.value());
  }
  auto output_size = op_infer::reduce_ops_npu_output_size(self, real_dim, keepdim);
  auto real_correction = correction.has_value() ? correction.value().toInt() : 1;

  at_npu::native::OpPreparation::check_tensor(
      {self},
      result,
      self,
      output_size);

  auto rd = at::IntArrayRef(real_dim);
  EXEC_NPU_CMD(aclnnVarCorrection, self, rd, real_correction, keepdim, result);
  return result;
}

at::Tensor& var_out(
    const at::Tensor& self,
    at::DimnameList dim,
    const c10::optional<c10::Scalar>& correction,
    bool keepdim,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnVarCorrection, acl_op::var_out(self, dim, correction, keepdim, result));
  return op_api::var_out(self, dimnames_to_positions(self, dim), correction, keepdim, result);
}

at::Tensor var(
    const at::Tensor & self,
    at::OptionalIntArrayRef dim,
    const c10::optional<c10::Scalar>& correction,
    bool keepdim) {
  DO_COMPATIBILITY(aclnnVarCorrection, acl_op::var(self, dim, correction, keepdim));
  c10::SmallVector<int64_t, op_infer::SIZE> real_dim = {};
  if (dim.has_value()) {
    real_dim = op_infer::array_to_small_vector(dim.value());
  }
  auto output_size = op_infer::reduce_ops_npu_output_size(self, real_dim, keepdim);
  auto real_correction = correction.has_value() ? correction.value().toInt() : 1;
  auto result = npu_preparation::apply_tensor_without_format(output_size, self.options());

  at_npu::native::OpPreparation::check_tensor(
      {self},
      result,
      self,
      output_size);

  auto rd = at::IntArrayRef(real_dim);
  EXEC_NPU_CMD(aclnnVarCorrection, self, rd, real_correction, keepdim, result);
  return result;
}

<<<<<<< HEAD
at::Tensor var(const at::Tensor& self, bool unbiased) {
  return at::var(self, c10::nullopt, int64_t{unbiased ? 1 : 0});
}

at::Tensor var(
    const at::Tensor & self,
    at::DimnameList dim,
    const c10::optional<c10::Scalar>& correction,
    bool keepdim) {
  DO_COMPATIBILITY(aclnnVarCorrection, acl_op::var(self, dim, correction, keepdim));
  return op_api::var(self, dimnames_to_positions(self, dim), correction, keepdim);
}

at::Tensor var(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool unbiased,
    bool keepdim) {
  return at::var(self, dim, c10::make_optional<c10::Scalar>(unbiased ? 1 : 0), keepdim);
}

at::Tensor var(
    const at::Tensor& self,
    at::DimnameList dim,
    bool unbiased,
    bool keepdim) {
  return op_api::var(self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

std::tuple<at::Tensor, at::Tensor> var_mean(
    const at::Tensor& self,
    at::DimnameList dim,
    bool unbiased,
    bool keepdim) {
  return op_api::var_mean(self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

=======
>>>>>>> 9d8d6f2 (remove aten not support backward.)
std::tuple<at::Tensor, at::Tensor> var_mean(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<c10::Scalar>& correction,
    bool keepdim) {
  c10::SmallVector<int64_t, N> real_dim = op_plugin::utils::get_dimlist_for_tensor(self);
  if (dim.has_value()) {
    real_dim = op_infer::array_to_small_vector(dim.value());
  }
  int64_t real_correction = correction.has_value() ? correction.value().toInt() : 1;
  auto output_size = op_infer::reduce_ops_npu_output_size(self, real_dim, keepdim);
  auto var = npu_preparation::apply_tensor_without_format(output_size, self.options());
  auto mean = npu_preparation::apply_tensor_without_format(output_size, self.options());

  auto rd = at::IntArrayRef(real_dim);
  EXEC_NPU_CMD(aclnnVarMean, self, rd, real_correction, keepdim, mean, var);
  return std::tuple<at::Tensor, at::Tensor>(mean, var);
}

<<<<<<< HEAD
std::tuple<at::Tensor, at::Tensor> var_mean(
    const at::Tensor& self,
    at::DimnameList dim,
    const c10::optional<c10::Scalar>& correction,
    bool keepdim) {
  return op_api::var_mean(self, dimnames_to_positions(self, dim), correction, keepdim);
}

std::tuple<at::Tensor, at::Tensor> var_mean(const at::Tensor& self, bool unbiased) {
  return at::var_mean(self, c10::nullopt, c10::make_optional<c10::Scalar>(unbiased ? 1 : 0));
}

std::tuple<at::Tensor, at::Tensor> var_mean(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool unbiased,
    bool keepdim) {
  return at::var_mean(self, dim, c10::make_optional<c10::Scalar>(unbiased ? 1 : 0), keepdim);
}

=======
>>>>>>> 9d8d6f2 (remove aten not support backward.)
} // namespace op_api
