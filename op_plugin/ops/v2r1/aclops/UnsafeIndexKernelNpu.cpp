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
#include <ATen/ops/_unsafe_index_native.h>

namespace acl_op {
<<<<<<<< HEAD:op_plugin/ops/v2r1/aclops/UnsafeIndexKernelNpu.cpp
at::Tensor _unsafe_index(const at::Tensor &self, const torch::List<c10::optional<at::Tensor>> &indices)
{
    return at::native::_unsafe_index(self, indices);
========
at::Tensor& scatter_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src,
    c10::string_view reduce,
    at::Tensor& result)
{
    TORCH_CHECK(false, "scatter.reduce_out is not supported.", OPS_ERROR(ErrCode::NOT_SUPPORT));
}

at::Tensor& scatter_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Scalar& value,
    c10::string_view reduce,
    at::Tensor& result)
{
    TORCH_CHECK(false, "scatter.value_reduce_out is not supported.", OPS_ERROR(ErrCode::NOT_SUPPORT));
>>>>>>>> 80fec4df (commit open):op_plugin/ops/v1r11/aclops/ScatterKernelNpu.cpp
}
} // namespace acl_op
