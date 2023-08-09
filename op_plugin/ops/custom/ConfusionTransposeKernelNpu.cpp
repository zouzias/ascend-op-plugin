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

namespace{
at::Tensor confusion_transpose_npu(
    const at::Tensor& self,
    at::IntArrayRef perm,
    at::IntArrayRef shape,
    bool transpose_first) {
  c10::SmallVector<int64_t, SIZE> output_size;
  if (transpose_first) {
    output_size = op_infer::array_to_small_vector(shape);
  } else {
    for (int i = 0; i < perm.size(); i++) {
      output_size.emplace_back(shape[perm[i]]);
    }
  }

  at::Tensor result = npu_preparation::apply_tensor(self, output_size);
  at_npu::native::OpCommand cmd;
  cmd.Name("ConfusionTransposeD")
      .Input(self)
      .Output(result)
      .Attr("perm", perm)
      .Attr("shape", shape)
      .Attr("transpose_first", transpose_first)
      .Run();

  return result;
}
} // namespace

at::Tensor npu_confusion_transpose_backward(
    const at::Tensor& grad,
    at::IntArrayRef perm,
    c10::SymIntArrayRef size,
    bool transpose_first) {
  auto shape = c10::asIntArrayRefUnchecked(size);

  c10::SmallVector<int64_t, SIZE> svec_shape;
  if (transpose_first) {
    svec_shape = op_infer::array_to_small_vector(shape);
  } else {
    for (int i = 0; i < perm.size(); i++) {
      svec_shape.emplace_back(shape[perm[i]]);
    }
  }
  std::vector<int64_t> vec_perm;
  int64_t perm_len = perm.size();
  int64_t temp_perm[perm_len] = {0};
  for (int64_t i = 0; i < perm_len; i++) {
    temp_perm[perm[i]] = i;
  }
  vec_perm = std::vector<int64_t>(temp_perm, temp_perm+perm_len);
  perm = at::IntArrayRef(vec_perm);
  at::Tensor result = npu_preparation::apply_tensor(grad, shape);

  at_npu::native::OpCommand cmd;
  cmd.Name("ConfusionTransposeD")
      .Input(grad)
      .Output(result)
      .Attr("perm", perm)
      .Attr("shape", svec_shape)
      .Attr("transpose_first", transpose_first)
      .Run();
  return result;
}

at::Tensor npu_confusion_transpose(const at::Tensor& self,
    at::IntArrayRef perm,
    at::IntArrayRef shape,
    bool transpose_first) {
  return confusion_transpose_npu(self, perm, shape, transpose_first);
}
} // namespace op_plugin
