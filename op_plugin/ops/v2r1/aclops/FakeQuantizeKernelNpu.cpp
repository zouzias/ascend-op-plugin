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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> fake_quantize_per_channel_affine_cachemask(
    const at::Tensor& self, const at::Tensor& scale, const at::Tensor& zero_point,
    int64_t axis, int64_t quant_min, int64_t quant_max) {
  TORCH_CHECK(zero_point.scalar_type() == at::ScalarType::Int || zero_point.scalar_type() == at::ScalarType::Float ||
              zero_point.scalar_type() == at::ScalarType::Half, "Zero-point must be Int32, Float or Half, found", zero_point.scalar_type());
  TORCH_CHECK(scale.dim() == 1, "scale should be a 1-D tensor");
  TORCH_CHECK(zero_point.dim() == 1, "zero point should be a 1-D tensor");
  TORCH_CHECK(scale.numel() == zero_point.numel(), "scale and zero_point need to have the same dimensions");
  TORCH_CHECK(scale.numel() == self.size(axis), "dimensions of scale and zero-point are not consistant with input tensor");
  TORCH_CHECK(quant_min <= quant_max, "`quant_min` should be less than or equal to `quant_max`.");
  if(!at::isFloatingType(zero_point.scalar_type())){
      TORCH_CHECK(at::min(zero_point).item().toInt() >= quant_min &&
                  at::max(zero_point).item().toInt() <= quant_max, "`zero_point` must be between `quant_min` and `quant_max`");
  }
  TORCH_CHECK(axis >= 0 && axis <= self.dim(), "`axis` must be between 0 and number of dimensions of input");
  
  at::Tensor out = npu_preparation::apply_tensor(self, self.sizes());
  at::Tensor mask = npu_preparation::apply_tensor(self.sizes(), self.options().dtype(at::kBool), self);

  at_npu::native::OpCommand cmd;
  cmd.Name("FakeQuantAffineCachemask")
      .Input(self)
      .Input(scale)
      .Input(zero_point)
      .Output(out)
      .Output(mask)
      .Attr("axis", axis)
      .Attr("quant_min", quant_min)
      .Attr("quant_max", quant_max)
      .Run();
  return std::tie(out, mask);
}

std::tuple<at::Tensor, at::Tensor> _fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
    const at::Tensor& self, const at::Tensor& scale, const at::Tensor& zero_point, const at::Tensor& fake_quant_enabled, int64_t quant_min, int64_t quant_max) {
  TORCH_CHECK(scale.numel() == 1, "a Tensor with", scale.numel(), " elements cannot be converted to Scalar");
  TORCH_CHECK(zero_point.numel() == 1, "a Tensor with", zero_point.numel(), " elements cannot be converted to Scalar");
  TORCH_CHECK(zero_point.scalar_type() == at::ScalarType::Int, "Zero-point must be Int32");
  TORCH_CHECK(fake_quant_enabled.numel() == 1, "a Tensor with", fake_quant_enabled.numel(), " elements cannot be converted to Scalar");
  TORCH_CHECK(quant_min <= quant_max, "`quant_min` should be less than or equal to `quant_max`.");

  if (fake_quant_enabled.item().toFloat() < 1.0) {
      at::Tensor output = self.clone();
      at::Tensor mask = npu_preparation::apply_tensor(self, self.options().dtype(at::kBool));
      mask.fill_(true);
      return std::tie(output, mask);
  }

  std::vector<int64_t> tensor_broadcast_size = {self.size(0)};
  at::Tensor scale_broadcast = scale.expand(tensor_broadcast_size).contiguous();
  at::Tensor zero_point_broadcast = zero_point.expand(tensor_broadcast_size).contiguous();
  at::Tensor out = npu_preparation::apply_tensor(self, self.sizes());
  at::Tensor mask = npu_preparation::apply_tensor(self.sizes(), self.options().dtype(at::kBool), self);
  int64_t axis = 0;

  at_npu::native::OpCommand cmd;
  cmd.Name("FakeQuantAffineCachemask")
      .Input(self)
      .Input(scale_broadcast)
      .Input(zero_point_broadcast)
      .Output(out)
      .Output(mask)
      .Attr("axis", axis)
      .Attr("quant_min", quant_min)
      .Attr("quant_max", quant_max)
      .Run();
  return std::tie(out, mask);
}
} // namespace acl_op
