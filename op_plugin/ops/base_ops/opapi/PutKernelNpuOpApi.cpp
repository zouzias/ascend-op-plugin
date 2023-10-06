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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {

at::Tensor &put_(at::Tensor &self, const at::Tensor &index, const at::Tensor &source, bool accumulate)
{
    DO_COMPATIBILITY(aclnnInplacePut, acl_op::put_(self, index, source, accumulate));

    at_npu::native::OpPreparation::check_memory({self, index, source}, {self});

    EXEC_NPU_CMD(aclnnInplacePut, self, index, source, accumulate);
    return self;
}
} // namespace op_api
