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
at::Tensor& logaddexp_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, const at::Tensor& other) {
  at::Tensor self_exp = npu_preparation::apply_tensor(self);
  at::Tensor other_exp = npu_preparation::apply_tensor(self);

  at_npu::native::OpCommand cmd_exp_1, cmd_exp_2, cmd_add, cmd_log;
  cmd_exp_1.Name("Exp")
      .Input(self)
      .Output(self_exp)
      .Run();

  cmd_exp_2.Name("Exp")
      .Input(other)
      .Output(other_exp)
      .Run();

  at::Tensor add_result = npu_preparation::apply_tensor(self);
  auto unified_result = npu_preparation::binary_op_check(add_result, self_exp, other_exp, true);

  cmd_add.Name("Add")
      .Expect(unified_result)
      .Input(self_exp)
      .Input(other_exp)
      .Output(add_result)
      .Run();

  cmd_log.Name("Log")
      .Input(add_result)
      .Output(result)
      .Attr("base", (float)-1)
      .Attr("scale", (float)1)
      .Attr("shift", (float)0)
      .Run();

  return result;
}
} // namespace

at::Tensor& logaddexp_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  npu_preparation::CheckOut(
      {self, other},
      result,
      self,
      output_size);

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    logaddexp_out_npu_nocheck(contiguous_result, self, other);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    logaddexp_out_npu_nocheck(result, self, other);
  }
  return result;
}

at::Tensor logaddexp(const at::Tensor& self, const at::Tensor& other) {
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  at::Tensor result = npu_preparation::apply_tensor(self, output_size);
  logaddexp_out_npu_nocheck(result, self, other);
  return result;
}
} // namespace acl_op
