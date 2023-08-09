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

#include "op_plugin/ops/OpInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_plugin {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& linalg_svdvals_out(const at::Tensor& A, c10::optional<c10::string_view> driver, at::Tensor & S) {
  // Dummies
  auto U = at::empty({0}, A.options());
  auto Vh = at::empty({0}, A.options());
  op_plugin::_linalg_svd_out(A, false, false, driver, U, S, Vh);
  return S;
}

at::Tensor linalg_svdvals(const at::Tensor& A, c10::optional<c10::string_view> driver) {
  auto U = at::empty({0}, A.options());
  auto Vh = at::empty({0}, A.options());
  auto sizes = A.sizes().vec();
  int64_t k = std::min(A.size(-2), A.size(-1));
  sizes.pop_back();
  sizes.end()[-1] = k;
  auto S = npu_preparation::ApplyTensor(A, sizes);
  S.fill_(0);

  op_plugin::_linalg_svd_out(A, false, false, driver, U, S, Vh);
  return S;
}
} // namespace op_plugin
