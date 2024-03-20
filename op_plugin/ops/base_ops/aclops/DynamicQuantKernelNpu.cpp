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

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_dynamic_quant(const at::Tensor &input)
{    
    at::SmallVector<int64_t, op_infer::SIZE> output_size = input.sizes();
    at::SmallVector<int64_t, op_infer::SIZE> scale_size = input.sizes();

    scale_size.pop_back();

    at::Tensor output = npu_preparation::apply_tensor_without_format(output_size, c10::dtype(c10::ScalarType::Char));
    at::Tensor scale = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Float));

    EXEC_NPU_CMD(aclnnDynamicQuant, input, output, scale);
    return std::make_tuple(output, scale);
}
}