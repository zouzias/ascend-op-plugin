// Copyright (c) 2024 Huawei Technologies Co., Ltd
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
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_anti_quant(const at::Tensor &x, const at::Tensor &scale, const c10::optional<at::Tensor> &offset,
                          c10::optional<at::ScalarType> dstDtype, c10::optional<at::ScalarType> srcDtype)
{
  auto inputDtype = x.dtype();
  if (inputDtype != at::ScalarType::Char) {
    TORCH_CHECK(false, "Input x must be Int8");
  }

  at::ScalarType srcType = at::ScalarType::Char;
  if (srcDtype.has_value()) {
    srcType = srcDtype.value();
    if (srcType != at::ScalarType::Char && srcType != at::ScalarType::QUInt4x2) {
      TORCH_CHECK(false, "srcDtype must be Int8 or Int4");
    }
  }

  at::Tensor offsetTensor(at::zeros_like(scale));
  if (offset.has_value()) {
    offsetTensor = offset.value();
  }

  at::ScalarType dstType = c10::value_or_else(dstDtype, [] {return at::ScalarType::Half;});

  // construct the output tensor of the NPU
  at::Tensor result = at_npu::native::OpPreparation::apply_tensor(x, x.options().dtype(dstType));

  bool sqrtMode = false;

  if (srcType != inputDtype) {
    at::Tensor xTensor(x);
    xTensor.toType(srcType);
    EXEC_NPU_CMD(aclnnAscendAntiQuant, xTensor, scale, offsetTensor, dstType, sqrtMode, result);
  } else {
    EXEC_NPU_CMD(aclnnAscendAntiQuant, x, scale, offsetTensor, dstType, sqrtMode, result);
  }

  return result;
}
} // namespace op_api
