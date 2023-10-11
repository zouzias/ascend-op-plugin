// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/KernelNpuOutputSize.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {

static c10::SmallVector<int64_t, op_infer::SIZE> get_output_size_of_matmul(const at::Tensor &tensor1,
                                                                           const at::Tensor &tensor2)
{
    c10::SmallVector<int64_t, op_infer::SIZE> output_size;
    auto dim_tensor1 = tensor1.dim();
    auto dim_tensor2 = tensor2.dim();

    TORCH_CHECK(dim_tensor1 > 0 && dim_tensor2 > 0, "matmul_tp got error dimentions: ", "(", dim_tensor1, ", ",
                dim_tensor2, ")");

    if (dim_tensor1 == 1 && dim_tensor2 == 1) {
        output_size = {};
    } else if (dim_tensor1 == 2 && dim_tensor2 == 1) {
        output_size = {tensor1.size(0)};
    } else if (dim_tensor1 == 1 && dim_tensor2 == 2) {
        output_size = {tensor2.size(1)};
    } else if (dim_tensor1 == 2 && dim_tensor2 == 2) {
        output_size = {tensor1.size(0), tensor2.size(1)};
    } else if (dim_tensor1 >= 3 && (dim_tensor2 == 1 || dim_tensor2 == 2)) {
        // t1:(N, n, m) * t2:(m, p)
        auto size1 = tensor1.sizes();
        auto tmp = c10::SmallVector<int64_t, op_infer::SIZE>{tensor2.size(0), 1};
        auto size2 = dim_tensor2 == 1 ? tmp : tensor2.sizes();
        output_size.insert(output_size.end(), size1.begin(), size1.end() - 1);
        if (dim_tensor2 > 1) {
        output_size.push_back(size2[dim_tensor2 - 1]);
        }
    } else if ((dim_tensor1 == 1 || dim_tensor1 == 2) && dim_tensor2 >= 3) {
        auto tmp = c10::SmallVector<int64_t, op_infer::SIZE>{1, tensor1.size(0)};
        auto size1 = dim_tensor1 == 1 ? tmp : tensor1.sizes();
        auto size2 = tensor2.sizes();
        output_size.insert(output_size.end(), size2.begin(), size2.end() - 2);
        if (dim_tensor1 > 1) {
        output_size.push_back(size1[0]);
        }
        output_size.push_back(size2[dim_tensor2 - 1]);
    } else if (dim_tensor1 >= 3 && dim_tensor2 >= 3) {
        // t1:(b1, n, m1) * t2:(x2, m2, p)
        int64_t n = tensor1.size(-2);
        at::IntArrayRef batch_tensor1(tensor1.sizes().data(), dim_tensor1 - 2);
        int64_t p = tensor2.size(-1);
        at::IntArrayRef batch_tensor2(tensor2.sizes().data(), dim_tensor2 - 2);
        std::vector<int64_t> expand_batch_portion = at::infer_size(batch_tensor1, batch_tensor2);
        c10::SmallVector<int64_t, op_infer::SIZE> output_expand_size(expand_batch_portion);
        output_expand_size.insert(output_expand_size.end(), {n, p});
        output_size = output_expand_size;
    } else {
        TORCH_CHECK(false, "matmul got error sizes: ", "(", dim_tensor1, ", ", dim_tensor2, ")");
    }

    return output_size;
}

static c10::SmallVector<int64_t, op_infer::SIZE> get_output_size_parallel(const at::Tensor &x1, const at::Tensor &x2,
                                                                          int64_t world_size,
                                                                          int64_t gather_index)
{
    const at::Tensor &gather_input = x1 if gather_index == 0 else x2;

    // 第一维度 * world_size -> new_tensor
    real_dim = gather_input.size(0) * world_size;
    const at::Tensor &real_x1 = gather_input if gather_index == 0 else x1;
    const at::Tensor &real_x2 = gather_input if gather_index == 1 else x2;

    return get_output_size_of_matmul(real_x1, real_x2)
}

at::Tensor all_gather_base_matmul(const at::Tensor &self, const at::Tensor &x2, const c10::optional<at::Tensor> &bias,
                                  int64_t hcomm_addr, int64_t world_size, int64_t gather_index = 0)
{
    auto output_size = get_output_size_parallel(self, x2, world_size, gather_index);
    auto result = at_npu::native::OpPreparation::apply_tensor_without_format(output_size, self.options());
    EXEC_NPU_CMD(aclnnAllGatherMatmul, self, x2, bias, hcomm_addr, gather_index, result);
    return result;
}
}  // namespace op_api
