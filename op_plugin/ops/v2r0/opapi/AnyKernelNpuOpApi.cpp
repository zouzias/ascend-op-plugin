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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& any_out(const at::Tensor& self, at::Tensor& result) {
  at::SmallVector<int64_t, op_infer::N> dim_list = op_plugin::utils::get_dimlist_for_tensor(self);
  bool keep_dim = false;
  
  // check result for return
  auto output_size = op_infer::reduce_ops_npu_output_size(self, dim_list, keep_dim);
  npu_preparation::check_tensor({self}, result, result, output_size);
  at::IntArrayRef dims(dim_list);
  EXEC_NPU_CMD(aclnnAny, self, dims, keep_dim, result);
  return result;
}
}  // namespace op_api
