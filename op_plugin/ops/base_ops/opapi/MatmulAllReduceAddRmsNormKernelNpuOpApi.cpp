// Copyright (c) 2024 Huawei Technologies Co., Ltd
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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> npu_mm_all_reduce_add_rms_norm(const at::Tensor &x1,
                                                                 const at::Tensor &x2,
                                                                 const at::Tensor& residual,
                                                                 const at::Tensor& gamma,
                                                                 c10::string_view hcom,
                                                                 c10::string_view reduce_op,
                                                                 double epsilon,
                                                                 const c10::optional<at::Tensor> &bias,
                                                                 const c10::optional<at::Tensor> &antiquant_scale,
                                                                 const c10::optional<at::Tensor> &antiquant_offset,
                                                                 const c10::optional<at::Tensor> &dequant_scale,
                                                                 int64_t antiquant_group_size,
                                                                 int64_t comm_turn)
{
    // size of last dim of output should be the same as size of last dim of x2
    auto output_size = op_infer::array_to_small_vector(residual.sizes());
    // a8w8: dtype of output should be half.
    auto output_dtype = residual.scalar_type();
    auto y = at_npu::native::OpPreparation::apply_tensor_without_format(output_size,
                                                                        residual.options().dtype(output_dtype));
    auto norm_out = at_npu::native::OpPreparation::apply_tensor_without_format(output_size,
                                                                               residual.options().dtype(output_dtype));
    
    char *reduce_op_ptr = const_cast<char *>(reduce_op.data());
    char *hcom_ptr = const_cast<char *>(hcom.data());
    const at::Tensor &bias_real = bias.value_or(at::Tensor());
    int64_t stream_mode = ACL_STOP_ON_FAILURE;
    // a8w8: x1\x2 kChar; a16w8: x2 kChar;
    if (!isIntegralType(x1.scalar_type()) && !isIntegralType(x2.scalar_type())) {
        EXEC_NPU_CMD(aclnnMatmulAllReduceAddRmsNorm, x1, x2, bias_real, residual, gamma, epsilon, hcom_ptr,
                     reduce_op_ptr, comm_turn, stream_mode, y, norm_out);
    }
    if (isIntegralType(x1.scalar_type()) && isIntegralType(x2.scalar_type())) {
        const at::Tensor &dequant_scale_real = dequant_scale.value_or(at::Tensor());
        EXEC_NPU_CMD(aclnnQuantMatmulAllReduceAddRmsNorm, x1, x2, bias_real, dequant_scale_real, residual, gamma,
                     epsilon, hcom_ptr, reduce_op_ptr, comm_turn, stream_mode, y, norm_out);
    }
    if (!isIntegralType(x1.scalar_type()) && isIntegralType(x2.scalar_type())) {
        const at::Tensor &antiquant_scale_real = antiquant_scale.value_or(at::Tensor());
        const at::Tensor &antiquant_offset_real = antiquant_offset.value_or(at::Tensor());
        EXEC_NPU_CMD(aclnnWeightQuantMatmulAllReduceAddRmsNorm, x1, x2, bias_real, antiquant_scale_real,
                     antiquant_offset_real, residual, gamma, epsilon, hcom_ptr, reduce_op_ptr, comm_turn,
                     stream_mode, antiquant_group_size, y, norm_out);
    }

    return std::tuple<at::Tensor, at::Tensor>(y, norm_out);
}
}  // namespace op_api