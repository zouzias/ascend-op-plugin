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

bool check_weight_dim(size_t weight_dim_num, size_t num_weight, size_t weight_dim_0, size_t num_group_list,
                      size_t sum_group_list)
{
    bool result = false;
    if (2 == weight_dim_num && num_weight == num_group_list) {
        result = true;
    } else if (3 == weight_dim_num && 1 == num_weight && weight_dim_0 == num_group_list) {
        result = true;
    } else if (2 == weight_dim_num && 1 == num_weight && weight_dim_0 == sum_group_list) {
        result = true;
    }
    return result;
}

void check_dims(int64_t split_item, size_t num_x, const at::TensorList &weight, size_t num_group_list,
                size_t sum_group_list)
{
    size_t num_weight = weight.size();
    TORCH_CHECK(num_x > 0 && num_weight > 0,
        "Neither x nor weight could be empty." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(IN_NOT_SPLIT_OUT_NOT_SPLIT == split_item || IN_NOT_SPLIT_OUT_SPLIT == split_item
        || IN_SPLIT_OUT_NOT_SPLIT == split_item || IN_SPLIT_OUT_SPLIT == split_item,
        "The given split_item [", split_item, "] is invalid, which must be one of 0/1/2/3" + OPS_ERROR(ErrCode::PARAM));
    if (IN_NOT_SPLIT_OUT_NOT_SPLIT == split_item || IN_NOT_SPLIT_OUT_SPLIT == split_item) {
        TORCH_CHECK(num_x == num_weight && 0 == num_group_list,
            "When split_item = 0 or 2, the num of x tensors must equal the num of weight tensors, "
            "and there is supposed not to be group_list input" + OPS_ERROR(ErrCode::PARAM));
    } else if (IN_SPLIT_OUT_NOT_SPLIT == split_item) {
        TORCH_CHECK(num_x == 1 && num_weight == num_group_list,
            "When split_item = 1, the num of x tensors must equal 1, "
            "and the num of weight tensors is supposed to equal the length of group_list" + OPS_ERROR(ErrCode::PARAM));
    } else if (IN_SPLIT_OUT_SPLIT == split_item) {
        size_t dim_num_weight = weight[0].sizes().size();
        size_t dim_0_weight = weight[0].sizes()[0];
        TORCH_CHECK(check_weight_dim(num_weight, dim_num_weight, dim_0_weight, num_group_list),
            "Invalid dim of weight. When split_item = 3, only the following three situations are allowed:"
            "(1) The tensor nums of weight equals the length of group_list; the dim num of each tensor equals 2. "
            "(2) There is one tensor in weight with a dim num of 3; its first dim equals the length of group_list. "
            "(3) There is one tensor in weight with a dim num of 2; its first dim equals the sum of group_list. "
            + OPS_ERROR(ErrCode::PARAM));
    }
}

void creat_new_tensor_multi_dim(std::vector<at::Tensor> &y, const at::Tensor &x_i, const at::Tensor &weight_i,
                                c10::TensorOptions options)
{
    auto x_sizes = x_i.sizes();
    std::vector<int64_t> y_sizes(x_sizes.begin(), x_sizes.end());
    y_sizes.at(x_sizes.size() - 1) = weight_i.sizes()[1];

    auto output_size = op_infer::array_to_small_vector(y_sizes);
    y.emplace_back(npu_preparation::apply_tensor_without_format(output_size, options));
}

void creat_new_tensor(std::vector<at::Tensor> &y, size_t dim_m, size_t dim_n, c10::TensorOptions options,
                      int64_t group_type_value, size_t num_group_list)
{
    auto output_size = (2 == group_type) ? op_infer::array_to_small_vector({num_group_list, dim_m, dim_n})
                                         : op_infer::array_to_small_vector({dim_m, dim_n});
    y.emplace_back(npu_preparation::apply_tensor_without_format(output_size, options));
}

// Motivation for adapting this interface for each Torch version separately:
// 1. Optional TensorList is only supported in Torch2.1 and later versions.
//    Thus, "Tensor[] bias" is used in Torch1.11 and Torch2.0, while
//    "Tensor[]? bias=None" is used in Torch2.1 and later versions.
// 2. Even if "Int[]? group_list=None" is used for all Torch versions, the
//    auto-generated data type for optional IntList group_list in Torch2.0
//    is different from those in Torch1.11, Torch2.1, and later versions.
std::vector<at::Tensor> npu_grouped_matmul(const at::TensorList x, const at::TensorList weight,
    const at::TensorList bias, const at::TensorList scale, const at::TensorList offset,
    const at::TensorList antiquant_scale, const at::TensorList antiquant_offset, at::OptionalIntArrayRef group_list,
    c10::optional<int64_t> split_item, c10::optional<int64_t> group_type, c10::optional<at::ScalarType> output_dtype)
{
    auto num_x = x.size();
    auto group_list_real = group_list.value_or(at::IntArrayRef{});
    auto num_group_list = group_list_real.size();
    int64_t split_item_value = split_item.value_or(0);
    int64_t group_type_value = group_type.value_or(-1);
    int64_t sum_group_list = 0;
    for (size_t k = 0; k < num_group_list; ++k) {
        sum_group_list += group_list_real[k];
    }
    check_dims(split_item_value, num_x, weight, num_group_list, sum_group_list);

    std::vector<at::Tensor> y;
    c10::TensorOptions options = x[0].options().dtype(output_dtype.value_or(x[0].scalar_type()));

    if (IN_NOT_SPLIT_OUT_NOT_SPLIT == split_item_value) {
        y.reserve(num_x);
        for (int i = 0; i < num_x; i++) {
            creat_new_tensor_multi_dim(y, x[i], weight[i], options);
        }
    } else if (IN_SPLIT_OUT_NOT_SPLIT == split_item_value) {
        y.reserve(num_group_list);
        creat_new_tensor(y, group_list_real[0], weight[0].sizes()[1], options, group_type_value, num_group_list);
        for (int i = 1; i < num_group_list; i++) {
            creat_new_tensor(y, group_list_real[i] - group_list_real[i - 1], weight[i].sizes()[1], options,
                             group_type_value, num_group_list);
        }
    } else if (IN_NOT_SPLIT_OUT_SPLIT == split_item_value) {
        size_t dim_m = 0;
        for (int i = 0; i < num_x; i++) {
            dim_m += x[i].sizes()[0];
        }
        creat_new_tensor(y, dim_m, weight[0].sizes()[1], options, group_type_value, num_group_list);
    } else if (IN_SPLIT_OUT_SPLIT == split_item_value) {
        size_t dim_num_weight = weight[0].sizes().size();
        creat_new_tensor(y, x[0].sizes()[0], weight[0].sizes()[dim_num_weight - 1], options, group_type_value,
                         num_group_list);
    }
    at::TensorList result = at::TensorList(y);

    EXEC_NPU_CMD(aclnnGroupedMatmul, x, weight, bias, scale, offset, antiquant_scale,
                 antiquant_offset, group_list_real, split_item_value, group_type_value, result);

    return y;
}
}  // namespace op_api

