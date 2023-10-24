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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

tensor_list npu_moe_gating_top_k_softmax(const at::Tensor &gating, const c10::optional<at::Tensor> &finished_opt, int64_t k)
{
    TORCH_CHECK(gating.dim() == 2 or gating.dim() == 3, "The gating should be 2D or 3D");
    TORCH_CHECK(
        gating.scalar_type() == at::kHalf || gating.scalar_type() == at::kFloat || gating.scalar_type() == at::kBFloat16,
        "float16, float32 or bfloat16 tensor expected but got a tensor with dtype: ",
        gating.scalar_type());

    auto gating_size = gating.sizes();
    TORCH_CHECK(k >= 0 and k <= gating_size[gating.dim() - 1],
                "The k should be in [0, ", gating_size[gating.dim() - 1], "]");
    const at::Tensor &finished = c10::value_or_else(finished_opt, [] { return at::Tensor(); });
    if (finished.defined()) {
        TORCH_CHECK(
            finished.scalar_type() == at::kBool,
            "bool tensor expected but got a tensor with dtype: ",
            finished.scalar_type());
        auto finished_size = finished.sizes();
        TORCH_CHECK((gating.dim() - 1) == finished.dim(), "gating dims shoud be largs finished dims than 1.");
        TORCH_CHECK(gating_size[0] == finished_size[0], "Input rows shoud be same.");
        if (gating.dim() == 3) {
            TORCH_CHECK(gating_size[1] == finished_size[1], "Input rows shoud be same.");
        }
    }

    at::Tensor out;
    at::Tensor indices_out;
    at::Tensor source_row_out;
    if (gating.dim() == 3) {
        out = npu_preparation::apply_tensor_without_format({gating_size[0], gating_size[1], k}, gating.options());
        indices_out = npu_preparation::apply_tensor_without_format({gating_size[0], gating_size[1], k},
                                                                   gating.options().dtype(at::kInt));
        source_row_out = npu_preparation::apply_tensor_without_format({gating_size[0], gating_size[1], k},
                                                                      gating.options().dtype(at::kInt));
    } else {
        out = npu_preparation::apply_tensor_without_format({gating_size[0], k}, gating.options());
        indices_out = npu_preparation::apply_tensor_without_format({gating_size[0], k}, gating.options().dtype(at::kInt));
        source_row_out = npu_preparation::apply_tensor_without_format({gating_size[0], k}, gating.options().dtype(at::kInt));
    }

    if (k == 0) {
        return std::tie(out, indices_out, source_row_out);
    }

    for (int32_t i = 0; i < gating.dim(); i++) {
        if (gating_size[i] == 0) {
            return std::tie(out, indices_out, source_row_out);
        }
    }

    EXEC_NPU_CMD(aclnnMoeGatingTopKSoftmax, gating, finished, k, out, indices_out, source_row_out);

    return std::tie(out, indices_out, source_row_out);
}

}  // namespace op_api