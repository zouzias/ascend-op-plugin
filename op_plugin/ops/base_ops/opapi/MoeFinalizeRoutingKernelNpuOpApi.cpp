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

#include <torch/csrc/autograd/custom_function.h>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_moe_finalize_routing(const at::Tensor& expanded_permuted_rows, const at::Tensor& skip1,
                                    const c10::optional<at::Tensor>& skip2,
                                    const at::Tensor& bias, const at::Tensor& scales,
                                    const at::Tensor& expanded_src_to_dst_row,
                                    const at::Tensor& expert_for_source_row)
{
    at::Tensor result = npu_preparation::apply_tensor_without_format(skip1);

    EXEC_NPU_CMD(aclnnMoeFinalizeRouting, expanded_permuted_rows, skip1, skip2, bias, scales,
                 expanded_src_to_dst_row, expert_for_source_row, result);

    return result;
}
} // namespace op_api
