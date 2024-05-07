// Copyright (c) 2023-2024 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
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
#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_fake_add_three(const at::Tensor &x1, const at::Tensor &x2)
{
    at::Tensor y = npu_preparation::apply_tensor(x1);
    at_npu::native::OpCommand cmd;
    cmd.Name("FakeAddThree")
        .Input(x1, "x1")
        .Input(x2, "x2")
        .Output(y, "y")
        .Run();
    return y;
}
} // namespace acl_op