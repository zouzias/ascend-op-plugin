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

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor _npu_silent_check(at::Tensor &input_grad, const at::Tensor &val, at::Tensor &pre_val, at::Tensor &min_val,
                             at::Tensor &max_val, const at::Tensor &val_counter, int64_t c_min_steps, double c_thresh_l1,
                             double c_coeff_l1, double c_thresh_l2, double c_coeff_l2)
{
    DO_COMPATIBILITY(aclnnSilentCheck, acl_op::_npu_silent_check(input_grad, val, pre_val, min_val, max_val, val_counter, c_min_steps, c_thresh_l1, c_coeff_l1, c_thresh_l2, c_coeff_l2));
    at::Tensor result = npu_preparation::apply_tensor_without_format(val_counter);
    int32_t c_min_steps_cast = static_cast<int32_t>(c_min_steps);
    float c_thresh_l1_cast = static_cast<float>(c_thresh_l1);
    float c_coeff_l1_cast = static_cast<float>(c_coeff_l1);
    float c_thresh_l2_cast = static_cast<float>(c_thresh_l2);
    float c_coeff_l2_cast = static_cast<float>(c_coeff_l2);
    EXEC_NPU_CMD(aclnnSilentCheck, val, input_grad, pre_val, min_val, max_val, val_counter, c_min_steps_cast, c_thresh_l1_cast, c_coeff_l1_cast, c_thresh_l2_cast, c_coeff_l2_cast, result);
    return result;
}
} // namespace op_api
