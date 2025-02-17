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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> batch_norm_gather_stats_with_counts(
    const at::Tensor& self, const at::Tensor& mean, const at::Tensor& invstd,
    const c10::optional<at::Tensor>& running_mean_opt, const c10::optional<at::Tensor>& running_var_opt,
    double momentum, double eps, const at::Tensor& counts) {
  DO_COMPATIBILITY(aclnnBatchNormGatherStatsWithCounts,
                   acl_op::batch_norm_gather_stats_with_counts(self, mean, invstd, running_mean_opt, running_var_opt,
                                                               momentum, eps, counts));
  auto data_type = at::kFloat;
  if (self.scalar_type() == mean.scalar_type() && self.scalar_type() == at::kHalf) {
    data_type = at::kHalf;
  }
  at::Tensor mean_all = npu_preparation::apply_tensor_without_format({self.size(1)}, self.options().dtype(data_type));
  at::Tensor invstd_all = npu_preparation::apply_tensor_without_format({self.size(1)}, self.options().dtype(data_type));
  EXEC_NPU_CMD(aclnnBatchNormGatherStatsWithCounts, self, mean, invstd, running_mean_opt, running_var_opt, momentum,
               eps, counts, mean_all, invstd_all);
  return std::make_tuple(mean_all, invstd_all);
}
}  // namespace op_api
