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

namespace op_api {

std::tuple<at::Tensor, at::Tensor> _aminmax(const at::Tensor &self,
                                            const int64_t dim,
                                            bool keepdim) {
  DO_COMPATIBILITY(aclnnAminmaxDim, acl_op::_aminmax(self, dim, keepdim));
  at::SmallVector<int64_t, SIZE> dims = {dim};
  auto output_size = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);
  auto min = at_npu::native::OpPreparation::apply_tensor_without_format(self, output_size);
  auto max = at_npu::native::OpPreparation::apply_tensor_without_format(self, output_size);
  EXEC_NPU_CMD(aclnnAminmaxDim, self, dim, keepdim, min, max);
  return std::tie(min, max);
}

std::tuple<at::Tensor, at::Tensor> _aminmax(const at::Tensor &self) {
  DO_COMPATIBILITY(aclnnAminmaxDim, acl_op::_aminmax(self));
  at::IntArrayRef dims = op_plugin::utils::get_dimlist_for_tensor(self);
  auto output_size = op_infer::reduce_ops_npu_output_size(self, dims, false);
  auto min = at_npu::native::OpPreparation::apply_tensor_without_format(self, output_size);
  auto max = at_npu::native::OpPreparation::apply_tensor_without_format(self, output_size);
  EXEC_NPU_CMD(aclnnAminmaxAll, self, min, max);
  return std::tie(min, max);
}

std::tuple<at::Tensor, at::Tensor> aminmax(const at::Tensor &self,
                                           c10::optional<int64_t> dim,
                                           bool keepdim) {
  at::IntArrayRef dims;
  if (dim.has_value()) {
    dims = dim.value();
  } else {
    dims = op_plugin::utils::get_dimlist_for_tensor(self);
  }
  auto output_size = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);
  auto min = at_npu::native::OpPreparation::apply_tensor_without_format(self, output_size);
  auto max = at_npu::native::OpPreparation::apply_tensor_without_format(self, output_size);
  EXEC_NPU_CMD(aclnnAminmax, self, dims, keepdim, min, max);
  return std::tie(min, max);
}

std::tuple<at::Tensor &, at::Tensor &> aminmax_out(const at::Tensor &self,
                                                   c10::optional<int64_t> dim,
                                                   bool keepdim,
                                                   at::Tensor &min,
                                                   at::Tensor &max) {
  DO_COMPATIBILITY(aclnnAminmax, acl_op::aminmax_out(self, dim, keepdim, min, max));
  at::IntArrayRef dims;
  if (dim.has_value()) {
    dims = dim.value();
  } else {
    dims = op_plugin::utils::get_dimlist_for_tensor(self);
  }
  auto output_size = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);
  at_npu::native::OpPreparation::check_tensor({self}, min, self.scalar_type(), output_size);
  at_npu::native::OpPreparation::check_tensor({self}, max, self.scalar_type(), output_size);
  EXEC_NPU_CMD(aclnnAminmax, self, dims, keepdim, min, max);
  return std::tie(min, max);
}

} // namespace op_api

