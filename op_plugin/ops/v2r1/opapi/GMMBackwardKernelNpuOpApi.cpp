// Copyright (c) 2024 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

void _foreach_transpose(at::TensorList tensorList, std::vector<at::Tensor> &tensors)
{
    for (int i = 0; i< tensorList.size(); i++) {
        at::Tensor tensor = tensorList[i].transpose(-1, -2).contiguous();
        tensors.emplace_back(tensor);
    }
}

// Motivation for adapting this interface for each Torch version separately:
// 1. Optional TensorList is only supported in Torch2.1 and later versions.
//    Thus, "Tensor[] bias" is used in Torch1.11 and Torch2.0, while
//    "Tensor[]? bias=None" is used in Torch2.1 and later versions.
// 2. Even if "Int[]? group_list=None" is used for all Torch versions, the
//    auto-generated data type for optional IntList group_list in Torch2.1
//    is different from those in Torch1.11 and Torch2.0.
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<at::Tensor>> npu_gmm_backward(const at::TensorList grad, 
                                                                                                       const at::TensorList x,
                                                                                                       const at::TensorList weight,
                                                                                                       c10::OptionalIntArrayRef group_list)
{
    auto num_x = x.size();
    auto num_w = weight.size();
    auto group_list_real = group_list.value_or(at::IntArrayRef{});
    auto num_group_list = group_list_real.size();
    
    std::vector<at::Tensor> xt;
    std::vector<at::Tensor> wt;

    c10::SmallVector<int64_t, SIZE> group_list_reals;
    group_list_reals.emplace_back(group_list_real[0]);
    for (int i = 1; 1 < num_group_list; i++) {
        group_list_reals.emplace_back(group_list_real[i] - group_list_real[i - 1]);
    }
    at::IntArrayRef _group_list_real = at::IntArrayRef(group_list_reals);

    std::vector<at::Tensor> x_splits = x[0].split(_group_list_real);
    at::TensorList x_split = x_splits;
    std::vector<at::Tensor> grad_splits = grad[0].split(_group_list_real);
    std::vector<at::Tensor> grad_split;
    for (int i = 0; i < grad_splits.size(); i++) {
        at::Tensor grad_tensor = grad_splits[i].contiguous();
        grad_split.emplace_back(grad_tensor);
    }
    at::TensorList grad_real = grad_split;

    _foreach_transpose(x_split, xt);
    _foreach_transpose(weight, wt);

    at::TensorList xt_real = at::TensorList(xt);
    at::TensorList wt_real = at::TensorList(wt);

    auto bias_real = at::TensorList();
    auto empty_group_list = at::IntArrayRef{};

    std::vector<at::Tensor> grad_contiv;
    for (int i = 0; i < grad.size(); i++) {
        grad_contiv.emplace_back(grad[i].contiguous());
    }
    
    std::vector<at::Tensor> dx = npu_gmm(grad_contiv, wt_real, bias_real, group_list_real, 3);
    std::vector<at::Tensor> dw = npu_gmm(xt_real, grad_real, bias_real, empty_group_list, 2);
    std::vector<at::Tensor> db;

    std::vector<at::Tensor> dw_output;
    for (int i = 0; i < num_w; i++) {
        at::Tensor dw_tensor = dw[i].reshape(weight[i].sizes());
        dw_output.emplace_back(dw_tensor);
    }

    return std::tie(dx, dw_output, db);
}
}  // namespace op_api

