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

#include <ATen/NamedTensorUtils.h>

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {

void index_copy_npu_par_check(
    const int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source,
    const at::Tensor& result,
    std::string func) {
  // todo：测试下这个是干啥的。
  int64_t new_dim = at::maybe_wrap_dim(dim, result.dim());
  // todo: index只能是0-dim或1-dim
  TORCH_CHECK_INDEX(index.dim() < 2, func, ": Index should have dimension 1 or 0 (got ", index.dim(), ")");

  int64_t num_indices = index.numel();
  // todo: 当source是0-dim时，index只能有一个数
  TORCH_CHECK_INDEX(!(source.dim() == 0 && num_indices != 1),
      func, ": When source is scalar, index should have one element (got ", num_indices, ")");
  // todo: 当source和result都不是0-dim时，他们dim应该相同。
  TORCH_CHECK_INDEX(!((source.dim() != result.dim()) && (source.dim() != 0 && result.dim() != 0)),
      func, ": When source and destination are not scalars, "
      "their dimensionality must match. Source dimensionality (",
      source.dim(), "), destination dimensionality (", result.dim(), ")");

  TORCH_CHECK_INDEX(index.scalar_type() == at::ScalarType::Long, func, ": Expected LongTensor for index");

  // Check that source and destination slices have the same size
  // todo:除了dim维度，其他维度尺寸都记录下来
  auto self_sliced_sizes = result.sizes().vec();
  if (self_sliced_sizes.size() > 0) {
    self_sliced_sizes.erase(self_sliced_sizes.begin() + new_dim);
  }
  auto source_sliced_sizes = source.sizes().vec();
  if (source_sliced_sizes.size() > 0) {
    source_sliced_sizes.erase(source_sliced_sizes.begin() + new_dim);
  }

  // todo: 校验其他维度数量不一样或对应位置尺寸不一样。
  TORCH_CHECK(
      !(self_sliced_sizes.size() != source_sliced_sizes.size() ||
          !std::equal(self_sliced_sizes.begin(), self_sliced_sizes.end(), source_sliced_sizes.begin())),
      func, ": Source/destination tensor must have same slice shapes.\n",
      "Destination slice shape: ", self_sliced_sizes, " at dimension ", new_dim,
      " and source slice shape: ", source_sliced_sizes, " at dimension 0.");
  // todo: 检验source非0-dim时，
  TORCH_CHECK_INDEX(source.dim() == 0 || num_indices == source.size(new_dim),
      func, ": Number of indices (", num_indices,
      ") should be equal to source.size(newDim) (", source.size(new_dim), ")");
  
  // 校验 index不越界
  auto boundary_index = result.dim() == 0 ? 0: self_sliced_sizes.begin() + new_dim;
  for (int64_t i = 0; i< num_indices; i++) {
    auto specifical_index = index.dim() == 0 ? index.item<int64_t>() : index[i].item<int64_t>();
    TORCH_CHECK(specifical_index <= boundary_index, func, ": index ",specifical_index,
                " is out of bounds for dimension ",boundary_index, "with size ",boundary_index + 1);
  }
}
} // namespace acl_op
