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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api
{
  c10::SmallVector<int64_t, SIZE> tome_unmerge_npu_output_size(const at::Tensor &atten_out,
                                                               const at::Tensor &ori_indice_a,
                                                               const at::Tensor &ori_indice_b)
  {
    int64_t batch = atten_out.size(0);
    int64_t seqlenA = ori_indice_a.size(1);
    int64_t seqlenB = ori_indice_b.size(1);
    int64_t hiddenSize = atten_out.size(2);

    c10::SmallVector<int64_t, SIZE> output_size;
    output_size = {batch, seqlenA + seqlenB, hiddenSize};

    return output_size;
  }

  at::Tensor npu_tome_unmerge(at::Tensor const&atten_out,
                              at::Tensor const&ori_indice_a,
                              at::Tensor const&ori_indice_b,
                              at::Tensor const&topk_indice,
                              at::Tensor const&arg_max,
                              c10::optional<double> top_r_rate)
  {
    float top_r_rate_value = top_r_rate.value_or(0.5);
    c10::SmallVector<int64_t, op_infer::SIZE> output_size = tome_unmerge_npu_output_size(
        atten_out, ori_indice_a, ori_indice_b);
    at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(atten_out, output_size);

    EXEC_NPU_CMD(aclnnTomeUnmerge, atten_out, ori_indice_a, ori_indice_b, topk_indice, arg_max, top_r_rate_value, result);

    return result;
  }

} // namespace op_api
