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
#include "op_plugin/utils/OpAdapter.h"
#include <ATen/ops/quantize_per_tensor.h>

namespace acl_op {
double q_scale(const at::Tensor& self)
{
    return at::native::q_scale_quant(self);
}

at::Tensor q_per_channel_scales(const at::Tensor& self)
{
    return at::native::q_per_channel_scales(self);
}

int64_t q_zero_point(const at::Tensor& self)
{
    return at::native::q_zero_point_quant(self);
}

at::Tensor q_per_channel_zero_points(const at::Tensor& self)
{
    at::Tensor result = at::native::q_per_channel_zero_points(self);
    return result.to(at::Device(at::kPrivateUse1));
}

int64_t q_per_channel_axis(const at::Tensor& self)
{
    return at::native::q_per_channel_axis(self);
}

at::QScheme qscheme(const at::Tensor& self)
{
    return at::native::qscheme_quant(self);
}

at::Tensor dequantize(const at::Tensor& self)
{
    return at::native::dequantize_quantized(self);
}
} // namespace acl_op
