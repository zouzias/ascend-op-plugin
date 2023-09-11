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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/KernelNpuOutputSize.h"

namespace op_api {
const int8_t ALLOW_FP32_DOWN_PRECISION = 1;
const int8_t KEEP_DTYPE = 0;

using npu_preparation = at_npu::native::OpPreparation;
static inline void matmul_implement_npu(at::Tensor &out,
                                        const at::Tensor &self,
                                        const at::Tensor &mat2) {
  // allow dicrease precision
  int8_t cube_math_type = ALLOW_FP32_DOWN_PRECISION;
  EXEC_NPU_CMD(aclnnMatmul, self, mat2, out, cube_math_type);
  return;
}

// if input was column-major, return grad as column-order for efficiency
static inline bool is_column_major(const at::Tensor &mat) {
  bool row_major = (mat.stride(-1) == 1 && mat.stride(-2) == mat.size(-1));
  return false == row_major;
}

at::Tensor matmul_mat1_backward(const at::Tensor& self, const at::Tensor& other,
                                const at::Tensor& grad_output) {
  /*mat1_grad = grad * mat2^T*/
  at::Tensor mat1 = self;
  at::Tensor mat2 = other;
  at::Tensor grad = grad_output;
  // strip mat: (1, 1, m, n)-> (m, n)
  while (mat1.dim() > 2 && mat1.size(0) == 1) {
    mat1 = mat1.squeeze(0);
  }
  // unsqueese: (5)*(5)^ -> (1*5)*(1,5)^
  if (mat2.dim() == 1) {
    mat2 = mat2.unsqueeze(-1);
    grad = grad.unsqueeze(-1);
  }
  if (mat1.dim() == 1) {
    mat1 = mat1.unsqueeze(0);
    grad = grad.unsqueeze(-2);
  }
  at::Tensor output;
  if (mat1.dim() == 2) { // mat2 is 2维，from mm
    // 先转置在后面一个tensor:先转置再合并k轴
    if (is_column_major(mat1)&&mat2.dim()==2) { // mat2 is 2维
      output = npu_preparation::ApplyTensorWithSizes(mat1.t().sizes(), grad.options());
      // 列主序, (mat2*grad^T)^T:
      grad = grad.transpose(-2, -1);
      grad = grad.reshape({-1, grad.size(-1)});
      mat2 = mat2.reshape({mat2.size(-2), -1}); // 列向连续，列向融合？？
      matmul_implement_npu(output, mat2, grad);
      output = output.t();
      output = output.reshape(self.sizes());
    }else {
      output = npu_preparation::ApplyTensorWithSizes(mat1.sizes(), grad.options());
      // grad * mat2^T:先转置再合并k轴
      mat2 = mat2.transpose(-2, -1);
      mat2 = mat2.reshape({-1, mat2.size(-1)});
      grad = grad.reshape({grad.size(-2), -1});
      matmul_implement_npu(output, grad, mat2);
      output = output.reshape(self.sizes());
    }
  } else { // bmm
    if (is_column_major(mat1)) { // (mat2*grad^T)^T:
      grad = grad.transpose(-2, -1);
      auto expend_sizes = op_infer::matmul_output_size(mat2, grad);
      output = npu_preparation::ApplyTensorWithSizes(expend_sizes, grad.options());
      matmul_implement_npu(output, mat2, grad);
      output = output.transpose(-2, -1);
    }else { // grad * mat2^T
      mat2 = mat2.transpose(-2, -1);
      auto expend_sizes = op_infer::matmul_output_size(grad, mat2);
      output = npu_preparation::ApplyTensorWithSizes(expend_sizes, grad.options());
      matmul_implement_npu(output, grad, mat2);
    }
  }
  return output;
}

at::Tensor matmul_mat2_backward(const at::Tensor& self, const at::Tensor& other,
                                const at::Tensor& grad_output) {
  /*mat2_grad = mat1^T * grad*/
  at::Tensor mat1 = self;
  at::Tensor mat2 = other;
  at::Tensor grad = grad_output;
  // strip mat: (1, 1, m, n)-> (m, n)
  while (mat2.dim() > 2 && mat2.size(0) == 1) {
    mat2 = mat2.squeeze(0);
  }
  // unsqueese: (5)*(5)^ -> (1*5)*(1,5)^
  if (mat2.dim() == 1) {
    mat2 = mat2.unsqueeze(-1);
    grad = grad.unsqueeze(-1);
  }
  if (mat1.dim() == 1) {
    mat1 = mat1.unsqueeze(0);
    grad = grad.unsqueeze(-2);
  }
  at::Tensor output;
  if (mat2.dim() == 2) { // mat2 is 2维，form mm
    if (is_column_major(mat2)) {
      output = npu_preparation::ApplyTensorWithSizes(mat2.t().sizes(), mat1.options());
      // 列主序, (grad^T*mat1)^T:
      grad = grad.reshape({-1, grad.size(-1)});
      mat1 = mat1.reshape({-1, mat1.size(-1)});
      grad = grad.transpose(-2, -1);
      matmul_implement_npu(output, grad, mat1);
      output = output.t();
      output = output.reshape(other.sizes());
    }else {
      // mat1^T * grad:先合并k轴再转置
      output = npu_preparation::ApplyTensorWithSizes(mat2.sizes(), mat1.options());
      mat1 = mat1.reshape({-1, mat1.size(-1)});
      grad = grad.reshape({-1, grad.size(-1)});
      mat1 = mat1.transpose(-2, -1);
      matmul_implement_npu(output, mat1, grad);
      output = output.reshape(other.sizes());
    }
  } else { // bmm
    if (is_column_major(mat2)){ // (grad^T*mat1)^T:
      grad = grad.transpose(-2, -1);
      auto expend_sizes = op_infer::matmul_output_size(grad, mat1);
      output = npu_preparation::ApplyTensorWithSizes(expend_sizes, mat1.options());
      matmul_implement_npu(output, grad, mat1);
      output = output.transpose(-2, -1);
    } else { // mat1^T * grad
      mat1 = mat1.transpose(-2, -1);
      auto expend_sizes = op_infer::matmul_output_size(mat1, grad);
      output = npu_preparation::ApplyTensorWithSizes(expend_sizes, mat1.options());
      matmul_implement_npu(output, mat1, grad);
    }
  }
  return output;
}

std::tuple<at::Tensor, at::Tensor> matmul_backward(const at::Tensor &grad,
                                                   const at::Tensor &self,
                                                   const at::Tensor &other,
                                                   std::array<bool, 2> grad_input_mask) {
  if (!grad.defined()) {
    return std::make_tuple(at::Tensor(), at::Tensor());
  }
  // backward mat1 and mat2 separately
  at::Tensor self_grad;
  at::Tensor other_grad;
  if (grad_input_mask[1]) {
    other_grad = matmul_mat2_backward(self, other, grad);
  }
  if (grad_input_mask[0]) {
    self_grad = matmul_mat1_backward(self, other, grad);
  }
  // strip added dim: (5,1)->(5)
  if (other.dim() == 1 && other_grad.size(-1) == 1) {
    other_grad = other_grad.squeeze(-1);
  }
  return std::make_tuple(self_grad, other_grad);
}

} // namespace op_api

