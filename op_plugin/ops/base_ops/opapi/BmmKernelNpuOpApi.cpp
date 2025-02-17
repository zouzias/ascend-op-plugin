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

#include <ATen/NamedTensorUtils.h>

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor &bmm_out(const at::Tensor &self, const at::Tensor &mat2, at::Tensor &result)
{
    DO_COMPATIBILITY(aclnnBatchMatMul, acl_op::bmm_out(self, mat2, result));
    auto output_size = {self.size(0), self.size(1), mat2.size(2)};
    npu_preparation::check_tensor({self, mat2}, result, self.scalar_type(), output_size);

    // cube_math_type, an enumeration value of type int8 that determines which calculation logic the CUBE unit should
    // use and functions such as hfloat32 can be enabled through this switch
    int cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowMatmulHF32());
    EXEC_NPU_CMD(aclnnBatchMatMul, self, mat2, result, cube_math_type);

    auto outnames = at::namedinference::compute_bmm_outnames(result, self, mat2);
    at::namedinference::propagate_names_if_nonempty(result, outnames);
    return result;
}

at::Tensor bmm(const at::Tensor &self, const at::Tensor &mat2)
{
    DO_COMPATIBILITY(aclnnBatchMatMul, acl_op::bmm(self, mat2));

    // calculate the output size
    auto output_size = {self.size(0), self.size(1), mat2.size(2)};

    // construct the output tensor of the NPU
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, self.options());

    // cube_math_type, an enumeration value of type int8 that determines which calculation logic the CUBE unit should
    // use and functions such as hfloat32 can be enabled through this switch
    int cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowMatmulHF32());
    EXEC_NPU_CMD(aclnnBatchMatMul, self, mat2, result, cube_math_type);

    auto outnames = at::namedinference::compute_bmm_outnames(result, self, mat2);
    at::namedinference::propagate_names_if_nonempty(result, outnames);
    return result;
}

} // namespace op_api
