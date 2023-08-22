// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2023, Facebook CORPORATION.
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
#include "op_plugin/utils/OpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_prompt_flash_attention(
    const at::Tensor &query, const at::Tensor &key,
    const at::Tensor &value, int64_t head_num, c10::string_view input_layout,
    const at::Tensor c10::optional<at::Tensor> &padding_mask,
    const at::Tensor c10::optional<at::Tensor> &atten_mask,
    at::IntArrayRef actual_seq_lengths,
    double scale, int64_t pre_tokens, int64_t next_tokens)
{
  // construct the output tensor of the NPU
  auto output = npu_preparation::apply_tensor_without_format(query);

  // convert str
  std::string input_layout_str = std::string(input_layout);
  char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());

  // dispatch hostAPI
  EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnPromptFlashAttention, query, key, value, padding_mask, atten_mask, actual_seq_lengths,
                               head_num, scale, pre_tokens, next_tokens, input_layout_ptr, output);
  return output;
}

at::Tensor npu_incre_flash_attention(
    const at::Tensor &query, const at::Tensor &key,
    const at::Tensor &value, int64_t head_num, c10::string_view input_layout,
    const at::Tensor c10::optional<at::Tensor> &padding_mask,
    const at::Tensor c10::optional<at::Tensor> &atten_mask,
    at::IntArrayRef actual_seq_lengths,
    double scale)
{
  // construct the output tensor of the NPU
  auto output = npu_preparation::apply_tensor_without_format(query);

  // convert str
  std::string input_layout_str = std::string(input_layout);
  char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());

  // dispatch hostAPI
  EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnIncreFlashAttention, query, key, value, padding_mask, atten_mask, actual_seq_lengths,
                               head_num, scale, input_layout_ptr, output);
  return output;
}
} // namespace op_api