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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& isneginf_out(const at::Tensor& self, at::Tensor& out) {
  DO_COMPATIBILITY(aclnnIsNegInf, acl_op::isneginf_out(self, out));
  // resize_ the output size when size of out and self don't match with each other.
  if (out.sizes() != self.sizes()) {
    auto output_size = op_infer::input_same_output_size(self);
    out.resize_(output_size);
  }
  // dispatch hostAPI
  EXEC_NPU_CMD(aclnnIsNegInf, self, out);
  return out;
}

} // namespace op_api
