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

at::Tensor& replication_pad2d_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, at::IntArrayRef padding) {
  TORCH_CHECK(padding.size() == 4, "padding size is expected to be 4");
  at::Tensor self_cp = self.dim() == 3 ? self.unsqueeze(0) : self;
  c10::SmallVector<int64_t, N> vector_int;
  c10::SmallVector<int64_t, N> paddings_vector = op_infer::array_to_small_vector(padding);
  TORCH_CHECK(self_cp.dim() != 0, "The self should not be empty");
  paddings_vector.resize(2 * self_cp.dim(), 0);
  for (int64_t i = paddings_vector.size(); i > 1; i -= 2) {
    vector_int.emplace_back(paddings_vector[i - 2]);
    vector_int.emplace_back(paddings_vector[i - 1]);
  }
  // constructs the attr of the NPUAttrDesc
  c10::SmallVector<int64_t, N> value_tensor = {(int64_t)0};
  at_npu::native::OpCommand cmd;
  cmd.Name("PadV3")
      .Input(self_cp)
      .Input(vector_int, at::kInt)
      .Input(value_tensor, self.scalar_type())
      .Output(result)
      .Attr("mode", (string)"edge")
      .Attr("paddings_contiguous", true)
      .Run();
  if (self.dim() == 3) {
    result.squeeze_(0);
  }
  return result;
}
} // namespace

at::Tensor& replication_pad2d_out(
    const at::Tensor& self,
    at::IntArrayRef padding,
    at::Tensor& result) {
  auto output_size = op_infer::replication_pad2d_npu_output_size(self, padding);
  npu_preparation::CheckOut(
      {self},
      result,
      self,
      output_size);
  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    replication_pad2d_out_npu_nocheck(contiguous_result, self, padding);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    replication_pad2d_out_npu_nocheck(result, self, padding);
  }
  return result;
}

at::Tensor replication_pad2d(const at::Tensor& self, at::IntArrayRef padding) {
  auto output_size = op_infer::replication_pad2d_npu_output_size(self, padding);
  at::Tensor result = npu_preparation::apply_tensor(self, output_size);
  replication_pad2d_out_npu_nocheck(result, self, padding);
  return result;
}

} // namespace acl_op
