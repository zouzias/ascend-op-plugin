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

namespace{
at::Tensor& celu_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, at::Scalar& alpha) {
  at_npu::native::OpCommand cmd;
  cmd.Name("CeluV2")
      .Input(self)
      .Output(result)
      .Attr("alpha", alpha)
      .Run();
  return result;
}

at::Tensor celu_npu_impl(const at::Tensor& self, at::Scalar& alpha) {
  at::Tensor result = npu_preparation::apply_tensor(self);
  celu_out_npu_nocheck(result, self, alpha);
  return result;
}

at::Tensor& celu_backward_out_npu(at::Tensor& grad_input, const at::Tensor& grad_output,
    at::Scalar& alpha, const at::Tensor& output) {
  at_npu::native::OpCommand cmd;
  cmd.Name("EluGradV2")
      .Input(grad_output)
      .Input(output)
      .Output(grad_input)
      .Attr("alpha", alpha)
      .Run();
  return grad_input;
}
} // namespace

at::Tensor celu_backward(const at::Tensor& grad_output, const at::Scalar& alpha, const at::Tensor& output) {
  std::cout << "----this is celu_backward" << std::endl;
  at::Tensor result = npu_preparation::apply_tensor(grad_output);
  celu_backward_out_npu(result, grad_output, alpha, output);
  return result;
}

at::Tensor celu(const at::Tensor& self, const at::Scalar& alpha) {
  return celu_npu_impl(self, alpha);
}

at::Tensor& celu_(at::Tensor& self, const at::Scalar& alpha) {
  if (!npu_utils::check_match(&self)) {
    at::Tensor contiguousSelf = npu_utils::format_contiguous(self);
    at::Tensor result = celu_npu_impl(contiguousSelf, alpha);
    npu_utils::format_fresh_view(self, result);
  } else {
    auto result = celu_npu_impl(self, alpha);
    self.copy_(result);
  }
  return self;
}
} // namespace op_plugin
