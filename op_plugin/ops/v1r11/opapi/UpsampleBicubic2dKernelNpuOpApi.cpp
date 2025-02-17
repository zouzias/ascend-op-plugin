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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/AclOpsInterface.h"

namespace op_api {

at::Tensor upsample_bicubic2d(const at::Tensor& self, c10::optional<at::IntArrayRef> output_size,
                              bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors)
{
    DO_COMPATIBILITY(aclnnUpsampleBicubic2d,
                     acl_op::upsample_bicubic2d(self, output_size, align_corners, scale_factors));
    auto osize = op_infer::upsample_infershape_with_scale(self.sizes(), output_size, scale_factors);
    auto scales_h = op_plugin::utils::get_scale_value(scale_factors, 0);
    auto scales_w = op_plugin::utils::get_scale_value(scale_factors, 1);
    at::Tensor result = op_api::upsample_bicubic2d(self, osize, align_corners, scales_h, scales_w);
    return result;
}

} // namespace op_api
