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

#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

c10::SmallVector<int64_t, SIZE> cat_npu_output_size(c10::SmallVector<at::Tensor, N>& tensors, int64_t dimension) {
  bool all_skipped = true;
  int64_t n_dims = 0;
  at::Tensor* not_skipped_tensor;
  auto num_inputs = tensors.size();
  auto should_skip = [](const at::Tensor* t) {
    return t->nbytes() == 0 && t->dim() == 1;
  };

  for (int i = 0; i < num_inputs; i++) {
    if (should_skip((at::Tensor*)&tensors[i])) {
      continue;
    }
    // found a non-empty tensor
    all_skipped = false;
    not_skipped_tensor = (at::Tensor*)&tensors[i];
    n_dims = not_skipped_tensor->dim();
    break;
  }

  if (all_skipped) {
    c10::SmallVector<int64_t, SIZE> size = {0};
    return size;
  }

  // Compute size of the result in the cat dimension
  int64_t cat_dim_size = 0;
  for (int i = 0; i < num_inputs; i++) {
    at::Tensor* tensor = (at::Tensor*)&tensors[i];
    if (should_skip(tensor)) {
      continue;
    }
    cat_dim_size += tensor->size(dimension);
  }

  c10::SmallVector<int64_t, SIZE> size;
  size.resize(n_dims);
  for (int dim = 0; dim < n_dims; dim++) {
    int64_t result_dim_size = not_skipped_tensor->size(dim);
    if (dim == dimension) {
      result_dim_size = cat_dim_size;
    }
    size[dim] = result_dim_size;
  }

  return size;
}
