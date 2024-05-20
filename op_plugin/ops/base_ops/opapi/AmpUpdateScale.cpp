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

at::Tensor& _amp_update_scale_(
  at::Tensor& current_scale,
  at::Tensor& growth_tracker,
  const at::Tensor& found_inf,
  double growth_factor,
  double backoff_factor,
  int64_t growth_interval,
)
{
  growth_factor = static_cast<float>(growth_factor);
  backoff_factor = static_cast<float>(backoff_factor);
  EXEC_NPU_CMD(aclnnAmpUpdateScale, current_scale, growth_tracker, found_inf, growth_factor, backoff_factor, growth_interval);
}

}  // namespace op_api
