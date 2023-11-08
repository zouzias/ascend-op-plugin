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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
std::tuple<at::Tensor, at::Tensor, at::Tensor> native_group_norm_npu_nocheck(
    at::Tensor& y,
    at::Tensor& mean,
    at::Tensor& rstd,
    const at::Tensor& X,
    const at::Tensor& gamma,
    const at::Tensor& beta,
    int64_t num_groups,
    double eps,
    int64_t C)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("GroupNormV2")
        .Input(X)
        .Input(gamma)
        .Input(beta)
        .Output(y)
        .Output(mean)
        .Output(rstd)
        .Attr("num_groups", num_groups)
        .Attr("eps", static_cast<float>(eps))
        .Attr("is_training", true)
        .Run();

    return std::make_tuple(y, mean, rstd);
}
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_group_norm(
    const at::Tensor& X,
    const c10::optional<at::Tensor>& gamma_opt,
    const c10::optional<at::Tensor>& beta_opt,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    double eps)
{
    at::Tensor result = npu_preparation::ApplyTensor(X);
    at::Tensor mean = npu_preparation::apply_tensor_with_format({N * group}, X.options(), ACL_FORMAT_ND);
    at::Tensor rstd = npu_preparation::apply_tensor_with_format({N * group}, X.options(), ACL_FORMAT_ND);

    const at::Tensor& gamma_ = c10::value_or_else(gamma_opt, [] {return at::Tensor();});
    const at::Tensor gamma = gamma_.defined() ? gamma_ : at::ones({C}, X.options());
    const at::Tensor& beta_ = c10::value_or_else(beta_opt, [] {return at::Tensor();});
    const at::Tensor beta = beta_.defined() ? beta_ : at::zeros({C}, X.options());

    return native_group_norm_npu_nocheck(result, mean, rstd, X, gamma, beta, group, eps, C);
}
} // namespace acl_op
