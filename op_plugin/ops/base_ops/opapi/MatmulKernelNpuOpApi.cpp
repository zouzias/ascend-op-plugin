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
using npu_preparation = at_npu::native::OpPreparation;

const int8_t ALLOW_FP32_DOWN_PRECISION = 1;
const int8_t KEEP_DTYPE = 0;

static inline void matmul_implement_npu(at::Tensor &out,
                                        const at::Tensor &self,
                                        const at::Tensor &mat2) {
  // allow dicrease precision
  int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowMatmulHF32());
  EXEC_NPU_CMD(aclnnMatmul, self, mat2, out, cube_math_type);
  return;
}

at::Tensor matmul_forward(const at::Tensor &self, const at::Tensor &mat2) {
  at::NoNamesGuard guard;
  auto output_size = op_infer::matmul_output_size(self, mat2);
  auto out = at_npu::native::OpPreparation::apply_tensor_without_format(output_size, self.options());
  matmul_implement_npu(out, self, mat2);
  return out;
}

at::Tensor matmul(const at::Tensor &tensor1, const at::Tensor &tensor2) {
  DO_COMPATIBILITY(aclnnMatmul, acl_op::matmul(tensor1, tensor2));
  auto maybe_outnames = at::namedinference::compute_matmul_outnames(tensor1, tensor2);
  auto result = matmul_forward(tensor1, tensor2);
  at::namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}

at::Tensor &matmul_out(const at::Tensor &tensor1, const at::Tensor &tensor2, at::Tensor &result) {
  DO_COMPATIBILITY(aclnnMatmul, acl_op::matmul_out(tensor1, tensor2, result));
  auto maybe_outnames = at::namedinference::compute_matmul_outnames(tensor1, tensor2);
  // matmul_out don't support backward
  auto output_size = op_infer::matmul_output_size(tensor1, tensor2);
  at_npu::native::OpPreparation::check_tensor({tensor1, tensor2}, result, tensor1, output_size);
  matmul_implement_npu(result, tensor1, tensor2);
  at::namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}

} // namespace op_api

