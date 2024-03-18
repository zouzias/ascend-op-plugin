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

at::ScalarType get_output_dtype(const at::Tensor &x1, const c10::optional<at::Tensor> &dequant_scale)
{
    auto output_dtype = x1.scalar_type() == at::kChar ? at::ScalarType::Half : x1.scalar_type();
    if (dequant_scale.has_value()) {
        const at::Tensor &dequant = dequant_scale.value();
        if (dequant.scalar_type() == at::kBFloat16) {
            output_dtype = at::kBFloat16;
        }
    }
    return output_dtype;
}

void check_params(const at::Tensor &x1, const at::Tensor &x2,
                  const at::Tensor &residual, const at::Tensor &gamma,
                  double epsilon,
                  const c10::optional<at::Tensor> &antiquant_scale,
                  const c10::optional<at::Tensor> &antiquant_offset,
                  const c10::optional<at::Tensor> &dequant_scale)
{
    // check shape: shape of x1:[m,k]/[b,m,k], shape of x2:[k,n], shape of residual:[b,m,n], shape of gamma:[b,m,n],
    // k_x1 == k_x2
    // (m)_x1 == (b*m)_residual, or（b_x1=b_residual, m_x1=m_residual）
    TORCH_CHECK((x1.dim() == 2 or x1.dim() == 3), "x1 needs to be 2D or 3D, but got: ", x1.dim(), "D",
                OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(x2.dim() == 2, "x2 needs to be 2D, but got: ", x2.dim(), "D", OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(residual.dim() == 3, "residual needs to be 3D, but got: ", residual.dim(), "D",
                OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(gamma.dim() == 3, "residual needs to be 3D, but got: ", gamma.dim(), "D", OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(x1.size(x1.dim() - 1) == x2.size(0), "K of x1 and x2 should be same, but they are x1_k: ",
                x1.size(x1.dim() - 1), ", x2_k: ", x2.size(0), OPS_ERROR(ErrCode::VALUE));
    if (x1.dim() == 2) {
        TORCH_CHECK(x1.size(0) == (residual.size(0) * residual.size(1)),
                    "b*m of residual and m of x1 should be same, but they are b_residual:", residual.size(0),
                    ", m_residual:", residual.size(1), ", m_x1:", x1.size(0), OPS_ERROR(ErrCode::VALUE));
    } else {
        TORCH_CHECK((x1.size(0) * x1.size(1)) == (residual.size(0) * residual.size(1)),
                    "b*m of residual and b*m of x1 should be same, but they are b_residual:", residual.size(0),
                    ", m_residual:", residual.size(1), ", b_x1:", x1.size(0), ", m_x1:", x1.size(1), OPS_ERROR(ErrCode::VALUE));
    }
    // check shape relationship: n_x2 == n_residual, b_residual== b_gamma, m_residual== m_gamma, n_residual== n_gamma
    TORCH_CHECK(x2.size(1) == residual.size(2),
                "n of X2 and residual should be same, but they are n_X2:", x2.size(1),
                ", n_residual:", residual.size(2), OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(residual.size(0) == gamma.size(0),
                "b of residual and gamma should be same, but they are b_residual:", residual.size(0),
                ", b_gamma:", gamma.size(0), OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(residual.size(1) == gamma.size(1),
                "m of residual and gamma should be same, but they are m_residual:", residual.size(1),
                ", m_gamma:", gamma.size(1), OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(residual.size(2) == gamma.size(2),
                "n of residual and gamma should be same, but they are n_residual:", residual.size(2),
                ", n_gamma:", gamma.size(2), OPS_ERROR(ErrCode::VALUE));
    // check m,k,n value in [1, 65535]
    if (x1.dim() == 2) {
        TORCH_CHECK(x1.size(0) >= 1 && x1.size(0) <= 65535, "m of x1 should be in [1,65535], but it is x1_m: ",
                    x1.size(0), OPS_ERROR(ErrCode::VALUE));
    } else {
        TORCH_CHECK((x1.size(0) * x1.size(1)) >= 1 && (x1.size(0) * x1.size(1)) <= 65535,
                    "b*m of x1 should be in [1,65535], but it is x1_b:",
                    x1.size(0), ", x1_m:", x1.size(1), OPS_ERROR(ErrCode::VALUE));
    }
    TORCH_CHECK(x1.size(x1.dim() - 1) >= 1 && x1.size(x1.dim() - 1) <= 65535, "k of x1 should be in [1,65535], but it is x1_k: ",
                x1.size(x1.dim() - 1), OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(x2.size(1) >= 1 && x2.size(1) <= 65535, "n of x2 should be in [1,65535], but it is x2_n: ",
                x2.size(1), OPS_ERROR(ErrCode::VALUE));
    
    // check parameters.
    // aclnn apis for MC2 share one torch_npu api, therefore, each aclnn api only accepts parameters
    // that will be used. Any unused parameter will be seen as illegal. The job must be done here in
    // torch_npu api.
    // A8W8: antiquantScale and antiquantOffset should be None.
    if (isIntegralType(x1.scalar_type()) && isIntegralType(x2.scalar_type())) {
        TORCH_CHECK(x1.scalar_type() == at::kChar, "x1 must be an int8 tensor for quant.", OPS_ERROR(ErrCode::TYPE));
        TORCH_CHECK(x2.scalar_type() == at::kChar, "x2 must be an int8 tensor for quant.", OPS_ERROR(ErrCode::TYPE));
        TORCH_CHECK((!antiquant_scale.has_value() && !antiquant_offset.has_value()),
                    "when both dtype of x1 and dtype of x2 are equal to int8, "
                    "antiquantScale, antiquantOffset should both be null", OPS_ERROR(ErrCode::TYPE));
    }
    // A16W8: dequantScale should be None.
    if (!isIntegralType(x1.scalar_type()) && isIntegralType(x2.scalar_type())) {
        TORCH_CHECK(x2.scalar_type() == at::kChar, "x2 must be an int8 tensor for weight quant.", OPS_ERROR(ErrCode::TYPE));
        TORCH_CHECK((!dequant_scale.has_value()),
                    "when only dtype of x2 is equal to int8, dequantScale should be null", OPS_ERROR(ErrCode::TYPE));
    }
    // MC2 without quantization. antiquantScale and antiquantOffset and dequantScale should be None.
    if (!isIntegralType(x1.scalar_type()) && !isIntegralType(x2.scalar_type())) {
        TORCH_CHECK((!antiquant_scale.has_value() && !antiquant_offset.has_value() && !dequant_scale.has_value()),
                    "when neither dtype of x1 or dtype of x2 is equal to int8, "
                    "antiquantScale, antiquantOffset and dequantScale should all be null", OPS_ERROR(ErrCode::TYPE));
    }
}

std::tuple<at::Tensor, at::Tensor> npu_mm_all_reduce_add_rms_norm_(const at::Tensor &x1,
                                                                  const at::Tensor &x2,
                                                                  const at::Tensor &residual,
                                                                  const at::Tensor &gamma,
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
    check_params(x1, x2, residual, gamma, epsilon, antiquant_scale, antiquant_offset, dequant_scale);
    // size of last dim of output should be the same as size of last dim of x2
    auto output_size = op_infer::array_to_small_vector(residual.sizes());
    // a8w8: dtype of output should be half.
    auto output_dtype = residual.scalar_type();
    auto norm_out = at_npu::native::OpPreparation::apply_tensor_without_format(output_size,
                                                                               residual.options().dtype(output_dtype));
    char *reduce_op_ptr = const_cast<char *>(reduce_op.data());
    char *hcom_ptr = const_cast<char *>(hcom.data());
    const at::Tensor &bias_real = bias.value_or(at::Tensor());
    int64_t stream_mode = ACL_STOP_ON_FAILURE;
    // a8w8: x1\x2 kChar; a16w8: x2 kChar;
    if (!isIntegralType(x1.scalar_type()) && !isIntegralType(x2.scalar_type())) {
        EXEC_NPU_CMD(aclnnInplaceMatmulAllReduceAddRmsNorm, x1, x2, bias_real, residual, gamma, epsilon, hcom_ptr,
                     reduce_op_ptr, comm_turn, stream_mode, norm_out);
    }
    if (isIntegralType(x1.scalar_type()) && isIntegralType(x2.scalar_type())) {
        const at::Tensor &dequant_scale_real = dequant_scale.value_or(at::Tensor());
        EXEC_NPU_CMD(aclnnInplaceQuantMatmulAllReduceAddRmsNorm, x1, x2, bias_real, dequant_scale_real, residual,
                     gamma, epsilon, hcom_ptr, reduce_op_ptr, comm_turn, stream_mode, norm_out);
    }
    if (!isIntegralType(x1.scalar_type()) && isIntegralType(x2.scalar_type())) {
        const at::Tensor &antiquant_scale_real = antiquant_scale.value_or(at::Tensor());
        const at::Tensor &antiquant_offset_real = antiquant_offset.value_or(at::Tensor());
        EXEC_NPU_CMD(aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm, x1, x2, bias_real, antiquant_scale_real,
                     antiquant_offset_real, residual, gamma, epsilon, hcom_ptr, reduce_op_ptr, comm_turn,
                     stream_mode, antiquant_group_size, norm_out);
    }
    return std::tuple<at::Tensor, at::Tensor>(residual, norm_out);
}
}  // namespace op_api