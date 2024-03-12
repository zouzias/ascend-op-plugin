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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

const static int INT64T_SIZE = 8;

// convert dim to non-negative value
static int64_t wrap_dim(const at::Tensor &self, c10::optional<int64_t> dim)
{
  int64_t real_dim = dim.value_or(0);
  return (real_dim < 0) ? (real_dim + self.dim()) : real_dim;
}

// check tensor repeats is valid
static bool check_tensor_repeats(const at::Tensor &self, const at::Tensor &repeats, c10::optional<int64_t> dim)
{
  if (repeats.dim() == 0) {
    return true;
  }
  if (repeats.dim() == 1) {
    if (dim.has_value()) {
      // with dim：check repeats is rank 1 with 1 element / rank 1 with (self.size(dim)) elements
      int64_t real_dim = wrap_dim(self, dim);
      if (repeats.size(0) == self.size(real_dim) || repeats.size(0) == 1) {
        return true;
      }
    } else {
      // without dim: check repeats is rank 0/ rank 1 with 1 element / rank 1 with (self.numel()) elements
      if (repeats.size(0) == self.numel() || repeats.size(0) == 1) {
        return true;
      }
    }
  }

  return false;
}

// check dim is in range [-self.dim(), self.dim()-1]
static bool check_dim_valid(const at::Tensor &self, c10::optional<int64_t> dim)
{
  int64_t real_dim = dim.value_or(0);
  int64_t self_dim = self.dim();
  int64_t dim_min = std::min(-self_dim, self_dim - 1);
  int64_t dim_max = std::max(-self_dim, self_dim - 1);
  return (dim_min <= real_dim && real_dim <= dim_max);
}

static at::Tensor apply_result_tensor(const at::Tensor &self, c10::SmallVector<int64_t, INT64T_SIZE> &output_shape,
    c10::optional<int64_t> dim, c10::optional<int64_t> output_size)
{
  int64_t cur_dim = wrap_dim(self, dim);
  int64_t output_size_expected = output_shape[cur_dim];
  if (output_size.has_value() && self.numel() != 0) {
    TORCH_CHECK(output_size_expected == output_size, "Allocated size does not match required size." + OPS_ERROR(ErrCode::PARAM));
  }
  at::Tensor result = npu_preparation::apply_tensor_without_format(self, output_shape);
  return result;
}

at::Tensor repeat_interleave_backward_symint(
    const at::Tensor& input_grad,
    const at::Tensor& self,
    c10::SymInt repeats,
    c10::optional<int64_t> dim,
    c10::optional<int64_t> output_size)
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
    int64_t grad_dim = input_grad.dim();
    if (dim < 0) {
        dim = dim + grad_dim;
    }
    at::SmallVector<int64_t, SIZE> result_shape = input_grad.sizes();
    result_shape[dim] = repeats.size(0);

    at::Tensor result = npu_preparation::apply_tensor_with_format(result_shape, input_grad.options(), ACL_FORMAT_ND);
    EXEC_NPU_CMD(aclnnRepeatInterleaveGrad, input_grad, repeats, dim, result);
    result = result.view(self.sizes());
    return result;
}
} // namespace op_api
