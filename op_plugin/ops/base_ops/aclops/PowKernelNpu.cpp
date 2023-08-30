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

namespace {
// pow.Tensor_Tensor_out
at::Tensor& pow_tensor_tensor_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, const at::Tensor& exp) {
  at_npu::native::OpCommand cmd;
  cmd.Name("Pow")
      .Input(self)
      .Input(exp)
      .Output(result)
      .Run();

  return result;
}

// pow.Tensor_Scalar_out
at::Tensor& pow_tensor_scalar_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, at::Scalar exp) {
  at_npu::native::OpCommand cmd;
  if (exp.toFloat() == 2.0) {
    cmd.Name("Square")
        .Input(self)
        .Output(result)
        .Run();
  } else {
    cmd.Name("Pow")
        .Input(self)
        .Input(exp, self.scalar_type())
        .Output(result)
        .Run();
  }
  return result;
}

// pow.Scalar_out
at::Tensor& pow_scalar_out_npu_nocheck(at::Tensor& result, at::Scalar self, const at::Tensor& exp) {
  at_npu::native::OpCommand cmd;
  cmd.Name("Pow")
      .Input(self, exp.scalar_type())
      .Input(exp)
      .Output(result)
      .Run();

  return result;
}
} // namespace

// pow.Tensor_Tensor_out
at::Tensor& pow_out(const at::Tensor& self, const at::Tensor& exp, at::Tensor& result) {
  if (npu_preparation::IsCPUScalar(exp)) {
    return acl_op::pow_out(self, exp.item(), result);
  } else if (npu_preparation::IsCPUScalar(self)) {
    return acl_op::pow_out(self.item(), exp, result);
  } else {
    TORCH_CHECK(self.device() == result.device() && exp.device() == result.device(),
        "Expected all tensors to be on the same device, but found at least two devices, ",
        self.device(), " and ", result.device());
    auto result_type = at::result_type(self, exp);
    TORCH_CHECK(result.scalar_type() == result_type,
        "result type ", result_type, " can't be cast to the desired output type ",  result.scalar_type());
    at::Tensor self_copy = (self.scalar_type() != result_type) ? 
        at_npu::native::custom_ops::npu_dtype_cast(self, result_type) : self;
    at::Tensor exp_copy = (exp.scalar_type() != result_type) ? 
        at_npu::native::custom_ops::npu_dtype_cast(exp, result_type) : exp;
    auto output_size = op_infer::broadcast_ops_npu_output_size(self_copy, exp_copy);
    npu_preparation::CheckOut(
        {self_copy, exp_copy},
        result,
        self_copy,
        output_size);

    if (!npu_utils::check_match(&result)) {
      at::Tensor contiguous_result = npu_utils::format_contiguous(result);
      pow_tensor_tensor_out_npu_nocheck(contiguous_result, self_copy, exp_copy);
      npu_utils::format_fresh_view(result, contiguous_result);
    } else {
      pow_tensor_tensor_out_npu_nocheck(result, self_copy, exp_copy);
    }
    return result;
  }
}

// pow.Tensor_Scalar_out
at::Tensor& pow_out(const at::Tensor& self, const at::Scalar& exp, at::Tensor& result) {
  TORCH_CHECK(self.device() == result.device(),
      "Expected all tensors to be on the same device, but found at least two devices, ",
      self.device(), " and ", result.device());
  auto result_type = at::result_type(self, exp);
  TORCH_CHECK(result.scalar_type() == result_type,
      "result type ", result_type, " can't be cast to the desired output type ",  result.scalar_type());
  at::Tensor self_copy = (self.scalar_type() != result_type) ? 
      at_npu::native::custom_ops::npu_dtype_cast(self, result_type) : self;
  npu_preparation::CheckOut(
      {self_copy},
      result,
      self_copy);

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    pow_tensor_scalar_out_npu_nocheck(contiguous_result, self_copy, exp);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    pow_tensor_scalar_out_npu_nocheck(result, self_copy, exp);
  }
  return result;
}

// pow.Scalar_out
at::Tensor& pow_out(const at::Scalar& self, const at::Tensor& exp, at::Tensor& result) {
  TORCH_CHECK(exp.device() == result.device(),
      "Expected all tensors to be on the same device, but found at least two devices, ",
      exp.device(), " and ", result.device());
  auto result_type = at::result_type(self, exp);
  TORCH_CHECK(result.scalar_type() == result_type,
      "result type ", result_type, " can't be cast to the desired output type ",  result.scalar_type());
  at::Tensor exp_copy = (exp.scalar_type() != result_type) ? 
      at_npu::native::custom_ops::npu_dtype_cast(exp, result_type) : exp;
  npu_preparation::CheckOut(
      {exp_copy},
      result,
      exp_copy);

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    pow_scalar_out_npu_nocheck(contiguous_result, self, exp_copy);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    pow_scalar_out_npu_nocheck(result, self, exp_copy);
  }
  return result;
}

at::Tensor pow(const at::Tensor& self, const at::Tensor& exp) {
  if (npu_preparation::IsCPUScalar(exp)) {
    return acl_op::pow(self, exp.item());
  } else if (npu_preparation::IsCPUScalar(self)) {
    return acl_op::pow(self.item(), exp);
  } else {
    auto result_type = at::result_type(self, exp);
    at::Tensor self_copy = (self.scalar_type() != result_type) ? 
        at_npu::native::custom_ops::npu_dtype_cast(self, result_type) : self;
    at::Tensor exp_copy = (exp.scalar_type() != result_type) ? 
        at_npu::native::custom_ops::npu_dtype_cast(exp, result_type) : exp;
    auto output_size = op_infer::broadcast_ops_npu_output_size(self_copy, exp_copy);
    at::Tensor result = npu_preparation::ApplyTensor(output_size, self.options().dtype(result_type), self_copy);
    pow_tensor_tensor_out_npu_nocheck(result, self_copy, exp_copy);
    return result;
  }
}

at::Tensor pow(const at::Tensor& self, const at::Scalar& exp) {
  auto result_type = at::result_type(self, exp);
  at::Tensor result = npu_preparation::ApplyTensor(self, self.options().dtype(result_type));
  at::Tensor self_copy = (self.scalar_type() != result_type) ? 
      at_npu::native::custom_ops::npu_dtype_cast(self, result_type) : self;
  pow_tensor_scalar_out_npu_nocheck(result, self_copy, exp);
  return result;
}

at::Tensor pow(const at::Scalar& self, const at::Tensor& exp) {
  auto result_type = at::result_type(exp, self);
  at::Tensor result = npu_preparation::ApplyTensor(exp, exp.options().dtype(result_type));
  at::Tensor exp_copy = (exp.scalar_type() != result_type) ? 
      at_npu::native::custom_ops::npu_dtype_cast(exp, result_type) : exp;
  pow_scalar_out_npu_nocheck(result, self, exp_copy);
  return result;
}

at::Tensor& pow_(at::Tensor& self, const at::Tensor& exp) {
  acl_op::pow_out(self, exp, self);
  return self;
}

at::Tensor& pow_(at::Tensor& self, const at::Scalar& exp) {
  acl_op::pow_out(self, exp, self);
  return self;
}
} // namespace at_npu
