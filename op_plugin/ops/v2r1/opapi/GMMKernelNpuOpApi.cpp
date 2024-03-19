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
const static int64_t IN_NOT_SPLIT_OUT_NOT_SPLIT = 0;
const static int64_t IN_SPLIT_OUT_NOT_SPLIT = 1;
const static int64_t IN_NOT_SPLIT_OUT_SPLIT = 2;
const static int64_t IN_SPLIT_OUT_SPLIT = 3;
using npu_preparation = at_npu::native::OpPreparation;

bool _check_mat2_dim(size_t num_mat2, size_t dim_num_mat2, size_t dim_0_mat2, size_t num_group_list,
                     size_t sum_group_list)
{
    bool result = false;
    if (2 == dim_num_mat2 && num_mat2 == num_group_list) {
        result = true;
    } else if (3 == dim_num_mat2 && 1 == num_mat2 && dim_0_mat2 == num_group_list) {
        result = true;
    } else if (2 == dim_num_mat2 && 1 == num_mat2 && dim_0_mat2 == sum_group_list) {
        result = true;
    }
    return result;
}

void _check_dims(int64_t split_item, size_t num_mat1, const at::TensorList &mat2, size_t num_group_list,
                 size_t sum_group_list)
{
    size_t num_mat2 = mat2.size();
    TORCH_CHECK(num_mat1 > 0 && num_mat2 > 0,
        "Neither x nor weight could be empty." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(IN_NOT_SPLIT_OUT_NOT_SPLIT == split_item || IN_NOT_SPLIT_OUT_SPLIT == split_item
        || IN_SPLIT_OUT_NOT_SPLIT == split_item || IN_SPLIT_OUT_SPLIT == split_item,
        "The given split_item [", split_item, "] is invalid, which must be one of 0/1/2/3" + OPS_ERROR(ErrCode::PARAM));
    if (IN_NOT_SPLIT_OUT_NOT_SPLIT == split_item || IN_NOT_SPLIT_OUT_SPLIT == split_item) {
        TORCH_CHECK(num_mat1 == num_mat2 && 0 == num_group_list,
            "When split_item = 0 or 2, the num of mat1 tensors must equal the num of mat2 tensors, "
            "and there is supposed not to be group_list input" + OPS_ERROR(ErrCode::PARAM));
    } else if (IN_SPLIT_OUT_NOT_SPLIT == split_item) {
        TORCH_CHECK(num_mat1 == 1 && num_mat2 == num_group_list,
            "When split_item = 1, the num of mat1 tensors must equal 1, "
            "and the num of mat2 tensors is supposed to equal the length of group_list" + OPS_ERROR(ErrCode::PARAM));
    } else if (IN_SPLIT_OUT_SPLIT == split_item) {
        size_t dim_num_mat2 = mat2[0].sizes().size();
        size_t dim_0_mat2 = mat2[0].sizes()[0];
        TORCH_CHECK(check_mat2_dim(num_mat2, dim_num_mat2, dim_0_mat2, num_group_list, sum_group_list),
            "Invalid dim of mat2. When split_item = 3, only the following three situations are allowed:"
            "(1) The tensor nums of mat2 equals the length of group_list; the dim num of each tensor equals 2. "
            "(2) There is one tensor in mat2 with a dim num of 3; its first dim equals the length of group_list. "
            "(3) There is one tensor in mat2 with a dim num of 2; its first dim equals the sum of group_list. "
            + OPS_ERROR(ErrCode::PARAM));
    }
}

void _creat_new_tensor_multi_dim(std::vector<at::Tensor> &y, const at::Tensor &mat1_i, const at::Tensor &mat2_i,
                                 c10::TensorOptions options)
{
    auto mat1_sizes = mat1_i.sizes();
    std::vector<int64_t> y_sizes(mat1_sizes.begin(), mat1_sizes.end());
    y_sizes.at(mat1_sizes.size() - 1) = mat2_i.sizes()[1];

    auto output_size = op_infer::array_to_small_vector(y_sizes);
    y.emplace_back(npu_preparation::apply_tensor_without_format(output_size, options));
}

void _creat_new_tensor(std::vector<at::Tensor> &y, size_t dim_m, size_t dim_n, c10::TensorOptions options,
                       int64_t group_type_value, size_t num_group_list)
{
    auto output_size = (2 == group_type_value) ? op_infer::array_to_small_vector({num_group_list, dim_m, dim_n})
                                               : op_infer::array_to_small_vector({dim_m, dim_n});
    y.emplace_back(npu_preparation::apply_tensor_without_format(output_size, options));
}

// Motivation for adapting this interface for each Torch version separately:
// 1. Optional TensorList is only supported in Torch2.1 and later versions.
//    Thus, "Tensor[] bias" is used in Torch1.11 and Torch2.0, while
//    "Tensor[]? bias=None" is used in Torch2.1 and later versions.
// 2. Even if "Int[]? group_list=None" is used for all Torch versions, the
//    auto-generated data type for optional IntList group_list in Torch2.1
//    is different from those in Torch1.11 and Torch2.0.
std::vector<at::Tensor> npu_gmm(const at::TensorList mat1,
                                const at::TensorList mat2,
                                const at::TensorList bias,
                                c10::OptionalIntArrayRef group_list,
                                c10::optional<int64_t> split_item,
                                c10::optional<int64_t> group_type)
{
    auto num_mat1 = mat1.size();
    auto num_mat2 = mat2.size();
    auto group_list_real = group_list.value_or(at::IntArrayRef{});
    auto num_group_list = group_list_real.size();
    int64_t split_item_value = split_item.value_or(0);
    int64_t group_type_value = group_type.value_or(-1);
    int64_t sum_group_list = 0;
    for (size_t k = 0; k < num_group_list; ++k) {
        sum_group_list += group_list_real[k];
    }

    check_dims(split_item_value, num_mat1, mat2, num_group_list, sum_group_list);

    std::vector<at::Tensor> y;
    c10::TensorOptions options = mat1[0].options().dtype(mat1[0].scalar_type());

    // Split mat2 when size of mat2 is 1 shape of mat2 is (b, k, n)
    std::vector<at::Tensor> mat2_split;
    at::TensorList mat2_real;
    if (num_mat2 == 1) {
        std::vector<at::Tensor> mat2_splits = mat2[0].split(1);
        for (int i = 0; i < mat2_splits.size(); i++) {
            at::Tensor tensor = mat2_splits[i].squeeze();
            if (!tensor.is_contiguous()) {
                tensor = tensor.contiguous();
            }
            mat2_split.emplace_back(tensor);
        }
        mat2_real = mat2_split;
    } else {
        mat2_real = mat2;
    }
    num_mat2 = mat2_real.size();

    if (IN_NOT_SPLIT_OUT_NOT_SPLIT == split_item_value) {
        y.reserve(num_mat1);
        for (int i = 0; i < num_mat1; i++) {
            _creat_new_tensor_multi_dim(y, mat1[i], mat2_real[i], options);
        }
    } else if (IN_SPLIT_OUT_NOT_SPLIT == split_item_value) {
        y.reserve(num_mat2);
        for (int i = 0; i < num_mat2; i++) {
            _creat_new_tensor(y, group_list_real[i], mat2_real[i].sizes()[1], options,
                              group_type_value, num_group_list);
        }
    } else if (IN_NOT_SPLIT_OUT_SPLIT == split_item_value) {
        size_t dim_m = 0;
        for (int i = 0; i < num_mat1; i++) {
            dim_m += mat1[i].sizes()[0];
        }
        _creat_new_tensor(y, dim_m, mat2_real[0].sizes()[1], options, group_type_value, num_group_list);
    } else if (IN_SPLIT_OUT_SPLIT == split_item_value) {
        size_t dim_num_mat2 = mat2_real[0].sizes().size();
        _creat_new_tensor(y, mat1[0].sizes()[0], mat2_real[0].sizes()[dim_num_mat2 - 1], options, group_type_value,
                          num_group_list);
    }

    at::TensorList result = at::TensorList(y);
    auto scale_real = scale.value_or(at::TensorList());
    auto offset_real = offset.value_or(at::TensorList());
    auto antiquant_scale_real = antiquant_scale.value_or(at::TensorList());
    auto antiquant_offset_real = antiquant_offset.value_or(at::TensorList());
    EXEC_NPU_CMD(aclnnGroupedMatmul, mat1, mat2_real, bias, scale_real, offset_real, antiquant_scale_real,
                 antiquant_offset, group_list_real, split_item_value, group_type_value, result);

    return y;
}
}  // namespace op_api

