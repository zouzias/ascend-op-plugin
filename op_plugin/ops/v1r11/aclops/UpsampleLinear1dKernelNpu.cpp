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

namespace acl_op {

at::Tensor upsample_linear1d(
    const at::Tensor& self,
    c10::optional<at::IntArrayRef> output_size,
    bool align_corners,
    c10::optional<at::ArrayRef<double>> scale_factors) {
    TORCH_CHECK(
        self.dim() == 3,
        "It is expected input_size equals to 3, but got size ",
        self.dim(), OPS_ERROR(ErrCode::PARAM));

  auto osize = op_infer::upsample_infershape_with_scale(self.sizes(), output_size, scale_factors);
  auto scales_w = op_plugin::utils::get_scale_value(scale_factors, 0);

  return acl_op::upsample_linear1d(self, osize, align_corners, scales_w);
}

} // namespace acl_op
