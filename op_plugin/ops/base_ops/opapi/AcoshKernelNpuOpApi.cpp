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
    using npu_preparation = at_npu::native::OpPreparation;

    at::Tensor& acosh_out(const at::Tensor& self, at::Tensor& result) {
        DO_COMPATIBILITY(aclnnAcosh, acl_op::acosh_out(self, result));
        TORCH_CHECK(!isIntegralType(result.scalar_type(), true), "result type ", toString(self.scalar_type()),
        " can't be cast to the desired output type ", toString(result.scalar_type()));
        npu_preparation::check_tensor({
            self
        }, result, result.scalar_type(), self.sizes());
        EXEC_NPU_CMD(aclnnAcosh, self, result);
        return result;
    }

    at::Tensor acosh(const at::Tensor& self) {
        DO_COMPATIBILITY(aclnnAcosh, acl_op::acosh(self));
        auto output_options = (isIntegralType(self.scalar_type(), true)) ?
        self.options().dtype(at::kFloat) : self.options();
        at::Tensor result = npu_preparation::apply_tensor_without_format(self.sizes(), output_options);
        EXEC_NPU_CMD(aclnnAcosh, self, result);
        return result;
    }

    at::Tensor& acosh_(at::Tensor& self) {
        DO_COMPATIBILITY(aclnnInplaceAcosh, acl_op::acosh_(self));
        TORCH_CHECK(!isIntegralType(self.scalar_type(), true),
        "result type Float can't be cast to the desired output type ", toString(self.scalar_type()));
        EXEC_NPU_CMD(aclnnInplaceAcosh, self);
        return self;
    }
}
// namespace op_api
