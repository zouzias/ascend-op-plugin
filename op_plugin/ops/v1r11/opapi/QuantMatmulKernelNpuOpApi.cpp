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
const static int MIN_DIM = 2;
const static int X_MAX_DIM = 8;

using npu_preparation = at_npu::native::OpPreparation;

void bias_shape_check(const at::Tensor &x1, const at::Tensor &x2, c10::optional<at::Tensor> bias) {
  auto x1_dim_num = x1.dim();
  auto x2_n_dim = x2.size(x2_dim_num - 1);
  auto bias_dim_num = bias.dim();
  TORCH_CHECK(bias_dim_num == 1 || bias_dim_num == 3, "The bias dim should be 1 or 3. but bias_dim_num is ",
              bias_dim_num);
  if (bias_dim_num == 1) {
    auto bias_first_dim = bias.size(0);
    TORCH_CHECK(bias_first_dim == x2_n_dim, "The bias first dim should equal to n.
                but bias_first_dim is ", bias_first_dim);
  } else if (bias_dim_num == 3) {
      TORCH_CHECK(x1_dim_num != 3 || x2_dim_num != 3, "x1 or x2 should have batch dim");
      auto bias_first_dim = bias.size(0);
      auto bias_second_dim = bias.size(1);
      auto bias_third_dim = bias.size(2);
      if (x1_dim_num == 3) {
        TORCH_CHECK(bias_first_dim == x2.size(0), "batch dim should match, but bias_first_dim is ", bias_first_dim);
        TORCH_CHECK(bias_second_dim == 1, "second dim of bias should be 1, but bias_second_dim is ", bias_second_dim);
        TORCH_CHECK(bias_third_dim == x2_n_dim, "third dim should be equal to n, but bias_third_dim is ",
                    bias_third_dim);
      } else {
        TORCH_CHECK(bias_first_dim == x2.size(0), "batch dim should match, but bias_first_dim is ", bias_first_dim);
        TORCH_CHECK(bias_second_dim == 1, "second dim of bias should be 1, but bias_second_dim is ", bias_second_dim);
        TORCH_CHECK(bias_third_dim == x2_n_dim, "third dim should be equal to n, but bias_third_dim is ",
                    bias_third_dim);
      }
    }
}

at::Tensor npu_quant_matmul(const at::Tensor &x1, const at::Tensor &x2, const at::Tensor &scale,
                            c10::optional<at::Tensor> offset, c10::optional<at::Tensor> bias)
{
    auto x1_dim_num = x1.dim();
    TORCH_CHECK(x1_dim_num >= MIN_DIM && x1_dim_num <= X_MAX_DIM, "x1 shape dims should be 2~8, but it is ",
                x1_dim_num);
    auto x2_dim_num = x2.dim();
    TORCH_CHECK(x2_dim_num >= MIN_DIM && x2_dim_num <= X_MAX_DIM, "x2 shape dims should be 2~8, but it is ",
                x2_dim_num);
    auto x1_k_dim = x1.size(x1_dim_num - 1);
    auto x2_n_dim = x2.size(x2_dim_num - 1);
    auto x2_k_dim = x2.size(x2_dim_num - 2);
    TORCH_CHECK(x1_k_dim == x2_k_dim, "The k of x1 and x2 should be equal. but x1_k_dim is ",
                x1_k_dim, ", x2_k_dim is ", x2_k_dim);

    auto output_size = op_infer::array_to_small_vector(x1.sizes());
    output_size[x1_dim_num - 1] = x2.size(x2_dim_num - 1);
    at::Tensor result = npu_preparation::apply_tensor_without_format(x1, output_size);

    auto scale_dim_num = scale.dim();
    TORCH_CHECK(scale_dim_num == 1, "The scale dim should be 1. but scale_dim_num is ", scale_dim_num);
    auto scale_first_dim = scale.size(0);
    TORCH_CHECK(scale_first_dim == 1 || scale_first_dim == x2_n_dim, "The scale dim should be 1 or n.
                but scale_first_dim is ", scale_first_dim);
    if (offset.has_value())
    {
      auto offset_dim_num = offset.dim();
      TORCH_CHECK(offset_dim_num == 1, "The scale dim should be 1. but scale_dim_num is ", offset_dim_num);
      auto offset_first_dim = scale.size(0);
      TORCH_CHECK(offset_first_dim == 1 || offset_first_dim == x2_n_dim, "The offset dim should be 1 or n.
                  but offset_first_dim is ", offset_first_dim);
    }
    const at::Tensor &offset_real = offset.value_or(at::Tensor());

    if (bias.has_value()) {
      bias_shape_check(x1, x2, bias);
    }
    const at::Tensor &bias_real = bias.value_or(at::Tensor());
    EXEC_NPU_CMD(aclnnQuantMatmul, x1, x2, scale, offset_real, bias_real, result);
    return result;
}
}  // namespace op_api