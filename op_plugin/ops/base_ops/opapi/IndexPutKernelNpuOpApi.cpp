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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/AdvancedIndex.h"
#include <chrono>
#include <string>
#include <torch/torch.h>

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor index_put(
    const at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices,
    const at::Tensor& value,
    bool accumulate) {
  DO_COMPATIBILITY(aclnnIndexPutImpl, acl_op::index_put(self, indices, value, accumulate));
  return self.clone(at::MemoryFormat::Contiguous).index_put_(indices, value, accumulate);
}

at::Tensor& index_put_(
    at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices,
    const at::Tensor& value,
    const bool accumulate) {
  DO_COMPATIBILITY(aclnnIndexPutImpl, acl_op::index_put_(self, indices, value, accumulate));
  return at::_index_put_impl_(self, indices, value, accumulate, false);
}

at::Tensor& _index_put_impl_(
    at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices,
    const at::Tensor& value,
    const bool accumulate,
    const bool unsafe) {
  DO_COMPATIBILITY(aclnnIndexPutImpl, acl_op::_index_put_impl_(self, indices, value, accumulate, unsafe));
  if (self.device().type() == at::kCPU) {
    return at::native::_index_put_impl_(self, indices, value, accumulate, unsafe);
  }

  auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  for (uint64_t i = 0; i < indices.size(); i++) {
    int size = (indices[i].defined() && !(indices[i].sizes().empty()))? indices[i].sizes()[0] : 0;
      std::string name = "indexput_ori_idx_" + std::to_string(i) + "_size" + std::to_string(size) + "_" + std::to_string(timestamp) + ".pt";
      torch::save(indices[i], name);
  }

  auto indices_after = op_plugin::AdvanceIndex::npu_expand_tensors(self, indices, true);
  std::vector<at::Tensor> all_defined_indices;
  at::SmallVector<int64_t, op_infer::N> zeroSize = {0};
  at::Tensor emptyTensor = npu_preparation::apply_tensor_without_format(self, zeroSize);
  for (int i = 0; i < static_cast<int>(indices_after.size()); i++) {
    if (indices_after[i].defined()) {
      all_defined_indices.emplace_back(indices_after[i]);
      continue;
    }
    all_defined_indices.emplace_back(emptyTensor);
  }

  for (auto &all_defined_indice : all_defined_indices) {
    if (all_defined_indice.device() != self.device()) {
      all_defined_indice = all_defined_indice.to(self.device());
    }
  }
  at::TensorList indices_tensor_list = all_defined_indices;
  if (self.numel() != 0 && value.numel() != 0) {
    EXEC_NPU_CMD(aclnnIndexPutImpl, self, indices_tensor_list, value, accumulate, unsafe);
  }
  return self;
}

} // namespace op_api

