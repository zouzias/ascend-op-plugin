// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
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

at::Tensor npu_gemm(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c, float alpha, float beta,
                    int transA, int transB, int cubeMathType)
{
    auto output_size = op_infer::broadcast_ops_npu_output_size(a, b);

    at::ScalarType result_type = at::native::result_type(a, b);

    // construct the output tensor of the NPU
    at::Tensor result =
        npu_preparation::apply_tensor_without_format(output_size, a.options().dtype(result_type));

    EXEC_NPU_CMD(aclnnGemm, a, b, c, beta, transA, transB, result, cubeMathType);
    return result;
}
} // namespace op_api
