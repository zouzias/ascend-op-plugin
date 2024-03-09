// Copyright (c) 2024 Huawei Technologies Co., Ltd
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
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor repeat_interleave_backward(const at::Tensor& input_grad, const at::Tensor& self, int64_t repeats,
    c10::optional<int64_t> dim, c10::optional<int64_t> output_size)
{
    if (!dim.has_value()) {
        dim = -1;
    }
    int64_t grad_dim = input_grad.dim();
    if (dim < 0) {
        dim = dim + grad_dim;
    }
    at::SmallVector<int64_t, SIZE> input_grad_new_shape;
    for (int64_t dim_index = 0; dim_index < grad_dim; dim_index++) {
        if (dim_index != dim) {
            input_grad_new_shape.emplace_back(input_grad.size(index));
        } else {
            input_grad_new_shape.emplace_back(input_grad.size(index) / repeats);
            input_grad_new_shape.emplace_back(repeats);
        }
    }
    auto input_grad_reshape = input_grad.view(input_grad_new_shape);
    auto result = input_grad_reshape.sum(dim + 1).view(self.sizes());
    return result;
}

at::Tensor repeat_interleave_backward(const at::Tensor& input_grad, const at::Tensor& self, const at::Tensor& repeats,
    c10::optional<int64_t> dim, c10::optional<int64_t> output_size)
{
    if (!dim.has_value()) {
        dim = -1;
    }

}

} // namespace op_api