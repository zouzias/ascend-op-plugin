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

#include <ATen/native/TypeProperties.h>
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
 
namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
 
// at::Tensor& cauchy_out(at::TensorList tensors, at::Dimname dim, at::Tensor& result) {
//   // DO_COMPATIBILITY(aclnnCat, acl_op::cat_out(tensors, dim, result));
//   // return at::cat_out(result, tensors, dimname_to_position(tensors[0], dim));
// }
 
at::Tensor& geometric_(at::Tensor& self, double p, c10::optional<at::Generator> gen) {
  // DO_COMPATIBILITY(aclnnCat, acl_op::cat(tensors, dim));
  // return at::cat(tensors, dimname_to_position(tensors[0], dim));
  // https://github.com/pytorch/pytorch/blob/c77a4a409654dbc0ac4a528c37873b0acb1be32d/aten/src/ATen/core/TransformationHelper.h
  // ::ceil(at::log(val) / at::log1p(-p))
  return op_api::ceil_(op_api::div_(op_api::log_(self), at::Scalar(std::log1p(-p))));
}
}  // namespace op_api