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

namespace acl_op {
    at_npu::native::DynamicInputRegFunc scatter_list_func = [](DyNumAndIndex num_and_index,
                                                               std::string op_name) -> ge::OperatorPtr
    {
        auto ge_op = std::make_shared<ge::op::Pack>(op_name.c_str());
        ge_op->create_dynamic_input_byindex_x(num_and_index.front().first, num_and_index.front().second);
        return ge_op;
    };

    at::Tensor scatter_update(
        const at::Tensor &self,
        const at::Tensor &indices,
        const at::Tensor &updates,
        int64_t axis)
    {
        at::Tensor result = self.clone();

        // Note:
        // The attribute 'reduce' of Scatter only supports setting it to 'update'.
        at_npu::native::OpCommand cmd;
        cmd.Name("Scatter")
            .Input(result)
            .Input(indices)
            .Input(updates)
            .Output(result)
            .Attr("reduce", (string) "update")
            .Attr("axis", axis)
            .Run();

        return result;
    }

at::Tensor &scatter_update_(
    at::Tensor &self,
    const at::Tensor &indices,
    const at::Tensor &updates,
    int64_t axis)
{
    // Note:
    // The attribute 'reduce' of Scatter only supports setting it to 'update'.
    at_npu::native::OpCommand cmd;
    cmd.Name("Scatter")
        .Input(self)
        .Input(indices)
        .Input(updates)
        .Output(self)
        .Attr("reduce", (string) "update")
        .Attr("axis", axis)
        .Run();

    return self;
}

at::TensorList npu_scatter_list(
    const at::TensorList &self,
    const at::Tensor &indices,
    const at::Tensor &updates,
    const c10::optional<at::Tensor> mask,
    int64_t axis)
{
    const at::Tensor &maskopt = c10::value_or_else(mask, []
                                                   { return at::Tensor(); });
    std::vector<at::Tensor> result;
    for (const at::Tensor &tensor : self)
    {
        result.push_back(tensor.clone());
    }
    at::TensorList result_ = at::TensorList(result);

    // Note:
    // The attribute 'reduce' of ScatterList only supports setting it to 'update'.
    at_npu::native::OpCommand cmd;
    cmd.Name("ScatterList").DynamicInputReg(scatter_list_func, {{dynamic_num, 0}});
    for (uint i = 0; i < dynamic_num; i++)
    {
        string input_name = "x" + std::to_string(i);
        cmd.Input(self[i], input_name);
    }
    cmd.Input(indices)
        .Input(updates);
    if (maskopt.defined())
    {
        cmd.Input(maskopt);
    }

    cmd.Output(result_)
        .Attr("reduce", (string) "update")
        .Attr("axis", axis)
        .Run();

    return result_;
}

at::TensorList &npu_scatter_list_(
    at::TensorList &self,
    const at::Tensor &indices,
    const at::Tensor &updates,
    const c10::optional<at::Tensor> mask,
    int64_t axis)
{
    const at::Tensor &maskopt = c10::value_or_else(mask, []
                                                   { return at::Tensor(); });
    // Note:
    // The attribute 'reduce' of ScatterList only supports setting it to 'update'.
    at_npu::native::OpCommand cmd;
    cmd.Name("ScatterList").DynamicInputReg(scatter_list_func, {{dynamic_num, 0}});
    for (uint i = 0; i < dynamic_num; i++)
    {
        string input_name = "x" + std::to_string(i);
        cmd.Input(self[i], input_name);
    }
    cmd.Input(indices)
        .Input(updates);
    if (maskopt.defined())
    {
        cmd.Input(maskopt);
    }

    cmd.Output(self)
        .Attr("reduce", (string) "update")
        .Attr("axis", axis)
        .Run();

    return self;
}

} // namespace acl_op
