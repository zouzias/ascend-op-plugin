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

#include "torch_npu/csrc/libs/library_npu.h"

class OpPluginTest : public testing::Test {
protected:
  static void SetUpTestSuite() {
    auto device = "npu:0";
    torch_npu::init_npu(device);
  }

  static void TearDownTestSuite() {
    torch_npu::finalize_npu();
  }
};
