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

c10::SmallVector<int64_t, SIZE> reflection_pad2d_npu_output_size(const at::Tensor& self, at::IntArrayRef padding) {
  int64_t N = self.dim() == 3 ? 1 : self.size(-4);
  int64_t C = self.size(-3);
  int64_t H = self.size(-2);
  int64_t W = self.size(-1);
  int64_t padding_l = padding[0];
  int64_t padding_r = padding[1];
  int64_t padding_t = padding[2];
  int64_t padding_b = padding[3];
  int64_t Ho = H + padding_t + padding_b;
  int64_t Wo = W + padding_l + padding_r;
  c10::SmallVector<int64_t, SIZE> output_size = {N, C, Ho, Wo};
  return output_size;
}

at::Tensor& reflection_pad2d_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::IntArrayRef padding) {
  TORCH_CHECK(padding.size() == 4, "padding size is expected to be 4");
  at::Tensor self_cp = self.dim() == 3 ? self.unsqueeze(0) : self;
  c10::SmallVector<int64_t, N> vector_int;
  c10::SmallVector<int64_t, N> paddings_vector = op_infer::array_to_small_vector(padding);
  paddings_vector.resize(2 * self_cp.dim(), 0);
  for (int64_t i = paddings_vector.size(); i > 1; i -= 2) {
    vector_int.emplace_back(paddings_vector[i - 2]);
    vector_int.emplace_back(paddings_vector[i - 1]);
  }
  c10::SmallVector<int64_t, N> value_tensor = {(int64_t)0};
  at_npu::native::OpCommand cmd;
  if(self.dtype() == at::kHalf) {
    cmd.Name("PadV3")
        .Input(self_cp)
        .Input(vector_int, at::kInt)
        .Input(value_tensor, self.scalar_type())
        .Output(result)
        .Attr("mode", (string)"reflect")
        .Attr("paddings_contiguous", true)
        .Run();
  } else {
    cmd.Name("MirrorPad")
        .Input(self_cp)
        .Input(vector_int, at::kInt)
        .Output(result)
        .Attr("mode", (string)"REFLECT")
        .Run();
  }
  if (self.dim() == 3) {
    result.squeeze_(0);
  }
  return result;
}
} // namespace

at::Tensor& reflection_pad2d_out(
    const at::Tensor& self,
    at::IntArrayRef padding,
    at::Tensor& result) {
  auto output_size = reflection_pad2d_npu_output_size(self, padding);
  npu_preparation::CheckOut(
      {self},
      result,
      self,
      output_size);
  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    reflection_pad2d_out_npu_nocheck(contiguous_result, self, padding);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    reflection_pad2d_out_npu_nocheck(result, self, padding);
  }
  return result;
}

at::Tensor reflection_pad2d(const at::Tensor& self, at::IntArrayRef padding) {
  auto output_size = reflection_pad2d_npu_output_size(self, padding);
  at::Tensor result = npu_preparation::ApplyTensor(self, output_size);
  reflection_pad2d_out_npu_nocheck(result, self, padding);
  return result;
}

} // namespace op_plugin
