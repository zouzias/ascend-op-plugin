// Copyright (c) 2023, Huawei Technologies.All rights reserved.
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

#ifndef __TORCH_NPU_OP_PLUGIN_UTILS_INNER_COMPUTE_OP_API__
#define __TORCH_NPU_OP_PLUGIN_UTILS_INNER_COMPUTE_OP_API__

#include <ATen/ATen.h>
#include <ATen/Tensor.h>

namespace op_api {
at::Tensor& sum_out_common_nocheck(const at::Tensor& self, at::IntArrayRef dim, bool keepdim,
                                   c10::optional<c10::ScalarType> dtype, at::Tensor& result);
at::Tensor sum_common_nocheck(const at::Tensor& self, at::IntArrayRef dim, bool keepdim,
                              c10::optional<c10::ScalarType> dtype);
} // namespace op_api
#endif
