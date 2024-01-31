// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {

at::Tensor onnx_max_pool(const at::Tensor &self, at::IntArrayRef kernelShape, at::IntArrayRef strides, int autoPad,
                         at::IntArrayRef pads, at::IntArrayRef dilations, int ceilMode, at::Tensor &result)
{
    int64_t autopads = static_cast<int64_t>(autoPad);
    int64_t ceilmodes = static_cast<int64_t>(ceilMode);
    EXEC_NPU_CMD(aclnnMaxPool, self, kernelShape, strides, autopads, pads, dilations, ceilmodes, result);
    return result;
}

} // namespace op_api
