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

int64_t get_approximate(c10::string_view approximate)
{
    if (approximate == "none") {
        return 0;
    } else if (approximate == "tanh") {
        return 1;
    } else {
        return 0;
    }
}

at::Tensor gelu(const at::Tensor& self, c10::string_view approximate) {
    DO_COMPATIBILITY(aclnnGeluV2, acl_op::gelu(self));
    int64_t approximate_mode = get_approximate(approximate);
    at::Tensor result = npu_preparation::apply_tensor_without_format(self);
    EXEC_NPU_CMD(aclnnGeluV2, self, approximate_mode, result);
    return result;
}
}  // namespace op_api
