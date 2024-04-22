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
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_ascend_quant(const at::Tensor &x, const at::Tensor &scale, const c10::optional<at::Tensor> &offset,
                            c10::optional<ScalarType> dst_dtype)
{
    at::ScalarType dst_type = c10::value_or_else(dst_dtype, [] {return at::ScalarType::Char;});
    if (dst_dtype != at::ScalarType::Char) {
        TORCH_CHECK(false, "dst_dtype must be Int8 " + OPS_ERROR(ErrCode::TYPE));
    }
    at::Tensor result = at_npu::native::OpPreparation::apply_tensor(x, x.options().dtype(dst_type));
    EXEC_NPU_CMD(aclnnAscendQuant, x, scale, offset, false, "round", dst_type, result);
    return result;
}
} // namespace op_api
 