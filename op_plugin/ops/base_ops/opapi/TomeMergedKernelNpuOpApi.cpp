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

namespace op_api
{
    using npu_preparation = at_npu::native::OpPreparation;
    const int64_t HEADS = 8;
    std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_tome_merge(at::Tensor token_a, at::Tensor token_b,
                                                                  at::Tensor token_indice, at::Tensor arg_max,
                                                                  c10::optional<double> top_rate)
    {
        int64_t batch = token_a.size(0);
        int64_t seq_len_a = token_a.size(1);
        int64_t hidden_size = token_a.size(2);
        int64_t seq_len_b = token_b.size(1);
        float top_rate_value = top_rate.value_or(0.5);
        int64_t topR = static_cast<int64_t>((seq_len_a + seq_len_b) * top_rate_value);

        at::SmallVector<int64_t, op_infer::SIZE> unmerge_token_a_size = {batch, seq_len_a - topR, hidden_size};
        at::SmallVector<int64_t, op_infer::SIZE> unmerge_token_b_size = {batch, HEADS, seq_len_b, hidden_size};
        at::SmallVector<int64_t, op_infer::SIZE> unreduce_count_size = {batch, HEADS, seq_len_b};

        at::Tensor unmerge_token_a = npu_preparation::apply_tensor_without_format(token_a, unmerge_token_a_size);
        at::Tensor unmerge_token_b = npu_preparation::apply_tensor_without_format(token_a, unmerge_token_b_size);
        at::Tensor unreduce_count = npu_preparation::apply_tensor_without_format(unreduce_count_size, c10::dtype(c10::ScalarType::Float));

        EXEC_NPU_CMD(aclnnTomeMerge, token_a, token_b, token_indice, arg_max, top_rate_value, unmerge_token_a, unmerge_token_b, unreduce_count);
        return std::make_tuple(unmerge_token_a, unmerge_token_b, unreduce_count);
    }

}