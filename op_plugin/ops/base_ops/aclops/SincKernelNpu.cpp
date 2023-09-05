// Copyright (c) 2020 Huawei Technologies Co., Ltd
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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {

at::Tensor& sinc_out(const at::Tensor& self, at::Tensor& result) {
  return at::sinc_out(self.to("cpu"), result.to("cpu")).to(self.device());
}

at::Tensor sinc(const at::Tensor& self) {
  return at::sinc(self.to("cpu")).to(self.device());
}

at::Tensor& sinc_(at::Tensor& self) {
  return at::sinc_(self.to("cpu")).to(self.device());
}

} // namespace acl_op