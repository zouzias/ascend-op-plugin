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
at::Tensor& glu_grad_npu_out_nocheck(at::Tensor &result, const at::Tensor &grad_output, const at::Tensor &self, int64_t dim) {
  auto chunkedInput = self.chunk(2, dim);
  at::Tensor firstHalf = chunkedInput[0];
  at::Tensor secondHalf = chunkedInput[1];

  secondHalf = secondHalf.sigmoid();
  at::Tensor gradFirst = secondHalf.mul(grad_output);
  at::Tensor gradSecond = firstHalf.mul(secondHalf).mul_(1-secondHalf).mul_(grad_output);
  result = at::cat({gradFirst, gradSecond}, dim);

  return result;
}
} // namespace

at::Tensor& glu_backward_out(const at::Tensor &grad_output, const at::Tensor &self, int64_t dim, at::Tensor &result) {
  auto output_size = op_infer::input_same_output_size(self);
  npu_preparation::CheckOut(
      {grad_output, self},
      result,
      grad_output,
      output_size);

  TORCH_CHECK(self.dim() > 0, "glu does not support 0-dimensional Tensors");
  auto wrap_dim = at::maybe_wrap_dim(dim, self.dim());
  const int64_t nIn = self.size(wrap_dim);
  TORCH_CHECK(nIn % 2 == 0, "Halving dimension must be even, but dimension ",
              wrap_dim, " is size ", nIn);

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    glu_grad_npu_out_nocheck(contiguous_result, grad_output, self, dim);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    glu_grad_npu_out_nocheck(result, grad_output, self, dim);
  }
  return result;
}

at::Tensor glu_backward(const at::Tensor &grad_output, const at::Tensor &self, int64_t dim) {
  TORCH_CHECK(self.dim() > 0, "glu does not support 0-dimensional Tensors");
  auto wrap_dim = at::maybe_wrap_dim(dim, self.dim());
  const int64_t nIn = self.size(wrap_dim);
  TORCH_CHECK(nIn % 2 == 0, "Halving dimension must be even, but dimension ",
              wrap_dim, " is size ", nIn);

  at::Tensor result = npu_preparation::ApplyTensor(self);
  glu_grad_npu_out_nocheck(result, grad_output, self, dim);
  return result;
}
}  // op_plugin
