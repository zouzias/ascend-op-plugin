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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> _unique(
    const at::Tensor& self,
    bool sorted,
    bool return_inverse) {
  DO_COMPATIBILITY(aclnnUnique, acl_op::_unique(self, sorted, return_inverse));
  at::Tensor y = npu_preparation::apply_tensor_without_format(self, self.numel());
  at::Tensor y_inverse = return_inverse 
                            ? npu_preparation::apply_tensor_without_format(self.sizes(),
                                                                           self.options().dtype(at::kLong)) 
                            : npu_preparation::apply_tensor_without_format({0}, self.options().dtype(at::kLong));

  static auto opApiFuncAddr = [](){
    auto ret = GetOpApiFuncAddr("aclGetViewShape");
    TORCH_CHECK(ret != nullptr);
    return ret;
  }();
  using aclGetViewShapeFunc = int (*)(const aclTensor* tensor, int64_t** view_dims, uint64_t* view_dims_num);
  auto aclGetViewShape = reinterpret_cast<aclGetViewShapeFunc>(opApiFuncAddr);
  sorted = true;
  auto npuAclParams = EXEC_NPU_CMD_SYNC(aclnnUnique, self, sorted, return_inverse, y, y_inverse);
  int64_t* view_dims = nullptr;
  uint64_t view_dim_num = 0;
  auto ret = aclGetViewShape(npuAclParams.Get<3>(), &view_dims, &view_dim_num);
  TORCH_CHECK(ret == 0, "aclGetViewShape failed.");
  c10::SmallVector<int64_t, SIZE> output_size(view_dims, view_dims + view_dim_num);
  y.resize_(output_size);
  delete view_dims;
  view_dims = nullptr;
  return std::tuple<at::Tensor, at::Tensor>(y, y_inverse);
}

} // namespace op_api

