// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
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

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_trans_quant_param(const at::Tensor &scale, c10::optional<at::Tensor> offset)
{
    auto scale_dim_num = scale.dim();
    TORCH_CHECK(scale_dim_num == 1, "The scale dim should be 1. but scale_dim_num is ", scale_dim_num);
    auto scale_first_dim = scale.size(0);
    TORCH_CHECK(scale_first_dim == 1,  "The scale dim should be 1. but scale_first_dim is ", scale_first_dim);

    auto output_size = op_infer::array_to_small_vector(scale.sizes());
    at::Tensor result = npu_preparation::apply_tensor_without_format(scale, output_size);
    if (offset.has_value())
    {
      auto offset_dim_num = offset.dim();
      TORCH_CHECK(offset_dim_num == 1, "The scale dim should be 1. but scale_dim_num is ", offset_dim_num);
      auto offset_first_dim = scale.size(0);
      TORCH_CHECK(offset_first_dim == 1, "The scale dim should be 1. but scale_first_dim is ", offset_first_dim);
    }
    const at::Tensor &offset_real = offset.value_or(at::Tensor());
    const at::Tensor &bias_real = bias.value_or(at::Tensor());
    EXEC_NPU_CMD(aclnnQuantParam, scale, offset_real, result);
    return result;
}
}  // namespace op_api