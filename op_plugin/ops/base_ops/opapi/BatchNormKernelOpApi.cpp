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

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm(const at::Tensor& self,
                                                                 const c10::optional<at::Tensor>& weight_opt,
                                                                 const c10::optional<at::Tensor>& bias_opt,
                                                                 const c10::optional<at::Tensor>& running_mean_opt,
                                                                 const c10::optional<at::Tensor>& running_var_opt,
                                                                 bool train, double momentum, double eps) {
  DO_COMPATIBILITY(aclnnBatchNorm, acl_op::native_batch_norm(self, weight_opt, bias_opt, running_mean_opt,
                                                             running_var_opt, train, momentum, eps));
  // construct the output tensor of the NPU
  at::Tensor result = npu_preparation::apply_tensor_without_format(self.sizes(), self.options());
  at::Tensor save_mean;
  at::Tensor save_invstd;
  if (train) {
    save_mean = npu_preparation::apply_tensor_without_format({self.size(1)}, self.options().dtype(at::kFloat));
    save_invstd = npu_preparation::apply_tensor_without_format({self.size(1)}, self.options().dtype(at::kFloat));
  } else {
    save_mean = at::empty({0}, self.options());
    save_invstd = at::empty({0}, self.options());
  }

  EXEC_NPU_CMD(aclnnBatchNorm, self, weight_opt, bias_opt, running_mean_opt, running_var_opt, train, momentum, eps,
               result, save_mean, save_invstd);
  return std::tie(result, save_mean, save_invstd);
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> native_batch_norm_out(
    const at::Tensor& self, const c10::optional<at::Tensor>& weight_opt, const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& running_mean_opt, const c10::optional<at::Tensor>& running_var_opt, bool train,
    double momentum, double eps, at::Tensor& out, at::Tensor& save_mean, at::Tensor& save_invstd) {
  DO_COMPATIBILITY(aclnnBatchNorm,
                   acl_op::native_batch_norm_out(self, weight_opt, bias_opt, running_mean_opt, running_var_opt, train,
                                                 momentum, eps, out, save_mean, save_invstd));

  EXEC_NPU_CMD(aclnnBatchNorm, self, weight_opt, bias_opt, running_mean_opt, running_var_opt, train, momentum, eps, out,
               save_mean, save_invstd);
  return std::tie(out, save_mean, save_invstd);
}
}  // namespace op_api
