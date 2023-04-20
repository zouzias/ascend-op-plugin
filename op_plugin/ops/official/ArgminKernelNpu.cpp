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

at::Tensor argmin(
    const at::Tensor& self, 
    c10::optional<int64_t> dim, 
    bool keepdim) {
  TORCH_CHECK(
      self.numel() > 0,
      "cannot perform reduction function argmin on a "
      "tensor with no elements because the operation does not have an identity");
  at::Tensor input = dim.has_value() ? self : self.reshape({-1});
  int64_t dim_value = dim.has_value() ? dim.value() : 0;
  bool keepdim_value = dim.has_value() ? keepdim : false;
  auto output_size = op_infer::reduce_ops_npu_output_size(input, dim_value, keepdim_value);
  at::Tensor result = npu_preparation::ApplyTensorWithFormat(
      output_size,
      self.options().dtype(at::kInt),
      ACL_FORMAT_ND);
  c10::SmallVector<int64_t, N> dim_vector = {dim_value};

  at_npu::native::OpCommand cmd;
  cmd.Name("ArgMin")
      .Input(input)
      .Input(dim_vector, at::kInt)
      .Output(result)
      .Attr("keep_dims", keepdim_value)
      .Run();
  result = op_plugin::npu_dtype_cast(result, at::kLong);
  return result;
}
} // namespace op_plugin
