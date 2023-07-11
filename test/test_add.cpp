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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "op_plugin/ops/OpInterface.h"
#include "test/test_utils/op_plugin_test.h"

TEST_F(OpPluginTest, Add) {
  auto device ="npu:0";
  auto input_x = torch::rand({2, 3, 5}, torch::dtype(torch::kFloat).requires_grad(true));
  auto input_y = torch::rand({2, 3, 5}, torch::dtype(torch::kFloat).requires_grad(true));

  auto cpu_output = torch::add(input_x, input_y, 1);
  auto npu_output = op_plugin::add(input_x.to(device), input_y.to(device), 1);

  ASSERT_TRUE(torch::allclose(npu_output.to("cpu"), cpu_output, 1e-7, 1e-5));
}
