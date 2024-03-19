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
                                                                                                       const at::TensorList mat1,
                                                                                                       const at::TensorList mat2,
                                                                                                       c10::OptionalIntArrayRef group_list)
{
    auto num_mat1 = mat1.size();
    auto num_mat2 = mat2.size();
    auto group_list_real = group_list.value_or(at::IntArrayRef{});
    auto num_group_list = group_list_real.size();
    
    std::vector<at::Tensor> mat1t;
    std::vector<at::Tensor> mat2t;

    std::vector<at::Tensor> mat1_splits = mat1[0].split(group_list_real);
    at::TensorList mat1_split = mat1_splits;
    std::vector<at::Tensor> grad_splits = grad[0].split(group_list_real);
    std::vector<at::Tensor> grad_split;
    for (int i = 0; i < grad_splits.size(); i++) {
        at::Tensor grad_tensor = grad_splits[i].contiguous();
        grad_split.emplace_back(grad_tensor);
    }
    at::TensorList grad_real = grad_split;

    _foreach_transpose(mat1_split, mat1t);
    _foreach_transpose(mat2, mat2t);

    at::TensorList mat1t_real = at::TensorList(mat1t);
    at::TensorList mat2t_real = at::TensorList(mat2t);

    auto bias_real = at::TensorList();
    auto empty_group_list = at::IntArrayRef{};

    std::vector<at::Tensor> grad_contiv;
    for (int i = 0; i < grad.size(); i++) {
        grad_contiv.emplace_back(grad[i].contiguous());
    }
    
    std::vector<at::Tensor> dmat1 = npu_gmm(grad_contiv, mat2t_real, bias_real, group_list_real, 3);
    std::vector<at::Tensor> dmat2 = npu_gmm(mat1t_real, grad_real, bias_real, empty_group_list, 2);
    std::vector<at::Tensor> db;

    std::vector<at::Tensor> dmat2_output;
    for (int i = 0; i < num_mat2; i++) {
        at::Tensor dmat2_tensor = dmat2[i].reshape(mat2[i].sizes());
        dmat2_output.emplace_back(dmat2_tensor);
    }

    return std::tie(dmat1, dmat2_output, db);
}
}  // namespace op_api

