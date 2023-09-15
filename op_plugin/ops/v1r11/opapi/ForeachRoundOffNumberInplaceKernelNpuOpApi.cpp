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
#include <ATen/native/ForeachUtils.h>

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
    void exec_npu_cmd(at::TensorList self, const char roundMode) {
        at::native::check_foreach_api_restrictions(self);
        at::ScalarType scalarType = self[0].scalar_type();
        if (scalarType == at::ScalarType::Byte
        || scalarType == at::ScalarType::Char
        || scalarType == at::ScalarType::Short
        || scalarType == at::ScalarType::Int
        || scalarType == at::ScalarType::Long) {
            return;
        }

        at::Tensor round_mode_scalar_tensor = at_npu::native::CalcuOpUtil::CopyScalarToDevice(
            roundMode, at::ScalarType::Char);
        // dispatch hostAPI
        EXEC_NPU_CMD(round_mode_scalar_tensor, self, round_mode_scalar_tensor);
    }

    void _foreach_ceil_(at::TensorList self) {
        exec_npu_cmd(self, (char)3);
    }

    void _foreach_floor_(at::TensorList self) {
        exec_npu_cmd(self, (char)2);
    }
} // namespace op_api