// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include <ATen/Tensor.h>
#include <c10/util/SmallVector.h>
#include <c10/util/Optional.h>
#include <ATen/NamedTensorUtils.h>

#include "op_plugin/ops/OpInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_plugin {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
at::Tensor& log_softmax_nocheck(at::Tensor& result, const at::Tensor& self, int64_t dim) {
  c10::SmallVector <int64_t, N> dimList = {dim};
  at_npu::native::OpCommand cmd;
  cmd.Name("LogSoftmaxV2")
     .Input(self)
     .Attr("axes", dimList)
     .Output(result)
     .Run();
  return result;
}

at::Tensor log_softmax_nocheck(
  at::Tensor& result,
  const at::Tensor& self,
  int64_t dim,
  c10::optional <c10::ScalarType> dtype) {
  c10::ScalarType dstType;
  if (dtype.has_value()) {
    dstType = dtype.value();
  } else if (result.defined()) {
    dstType = result.scalar_type();
  } else {
    dstType = self.scalar_type();
  }

  // dtype same
  if (dstType == self.scalar_type()) {
    log_softmax_nocheck(result, self, dim);
    return result;
  }

  log_softmax_nocheck(result, self.toType(dstType), dim);
  return result;
}
}

at::Tensor log_softmax(
  const at::Tensor& self,
  int64_t dim,
  c10::optional <c10::ScalarType> dtype) {
  c10::ScalarType dstType = dtype.has_value() ? dtype.value() : self.scalar_type();

  if (dstType == self.scalar_type()) {
    return at::_log_softmax(self, dim, false);
  }

  return at::_log_softmax(self.toType(dstType), dim, false);
}

at::Tensor log_softmax(
  const at::Tensor& self,
  at::Dimname dim,
  c10::optional <c10::ScalarType> dtype) {
  return op_plugin::log_softmax(self, dimname_to_position(self, dim), dtype);
}

at::Tensor _log_softmax(const at::Tensor& self, int64_t dim, bool half_to_float) {
  // construct the output tensor of the NPU
  at::Tensor result;
  if (half_to_float) {
    result = npu_preparation::ApplyTensor(self, self.options()
                                                    .dtype(c10::ScalarType::Float));
  } else {
    result = npu_preparation::ApplyTensor(self);
  }

  // calculate the output result of the NPU
  log_softmax_nocheck(result, self, dim, result.scalar_type());

  return result;
}
} // namespace op_plugin
