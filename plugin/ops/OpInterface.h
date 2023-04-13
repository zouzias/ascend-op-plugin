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

#ifndef __TORCH_NPU_OP_PLUGIN_INTERFACE__
#define __TORCH_NPU_OP_PLUGIN_INTERFACE__

#include <ATen/Tensor.h>
#include <ATen/ATen.h>

namespace op_plugin {
// Abs
at::Tensor& abs_out(const at::Tensor& self, at::Tensor& result);
at::Tensor abs(const at::Tensor& self);
at::Tensor& abs_(at::Tensor& self);

// __and__
at::Tensor __and__(const at::Tensor& self, const at::Tensor& other);
at::Tensor __and__(const at::Tensor& self, const at::Scalar& other);

// fill_
at::Tensor& fill_(at::Tensor& self, const at::Tensor& other);
at::Tensor& fill_(at::Tensor& self, const at::Scalar& other);

// __ilshift__
at::Tensor& __ilshift__(at::Tensor& self, const at::Tensor& other);
at::Tensor& __ilshift__(at::Tensor& self, const at::Scalar& other);

// __irshift__
at::Tensor& __irshift__(at::Tensor& self, const at::Tensor& other);
at::Tensor& __irshift__(at::Tensor& self, const at::Scalar& other);

// __lshift__
at::Tensor __lshift__(const at::Tensor& self, const at::Tensor& other);
at::Tensor __lshift__(const at::Tensor& self, const at::Scalar& other);

// __rshift__
at::Tensor __rshift__(const at::Tensor& self, const at::Tensor& other);
at::Tensor __rshift__(const at::Tensor& self, const at::Scalar& other);

// __ior__
at::Tensor& __ior__(at::Tensor& self, const at::Tensor& other);
at::Tensor& __ior__(at::Tensor& self, const at::Scalar& other);

// __xor__
at::Tensor __xor__(const at::Tensor& self, const at::Tensor& other);
at::Tensor __xor__(const at::Tensor& self, const at::Scalar& other);

//__or__
at::Tensor __or__(const at::Tensor& self, const at::Tensor& other);
at::Tensor __or__(const at::Tensor& self, const at::Scalar& other);

// _unique2
tuple<at::Tensor, at::Tensor, at::Tensor> _unique2(
    const at::Tensor& self_op,
    bool sorted,
    bool return_inverse,
    bool return_counts);
}  // namespace op_plugin
#endif