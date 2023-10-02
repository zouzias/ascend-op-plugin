// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
    const static int FLASH_THRESHOLD = 512;
    using namespace at_npu::native;

    enum class DropOutStatus {
        DROPOUT_NORMAL = 0,
        DROPOUT_NONE,
        DROPOUT_ALL
    };

    namespace {
        DropOutStatus get_dropout_status(double keep_prob)
        {
            if (keep_prob == 0) {
                return DropOutStatus::DROPOUT_ALL;
            }
            if (keep_prob == 1.) {
                return DropOutStatus::DROPOUT_NONE;
            }
            return DropOutStatus::DROPOUT_NORMAL;
        }

        at::Tensor format_trans(const at::Tensor & at_tensor)
        {
            if (at_tensor.defined()) {
                TORCH_CHECK(torch_npu::utils::is_npu(at_tensor), "only npu tensor is supported");
                return custom_ops::npu_format_cast(at_tensor, ACL_FORMAT_ND);
            }
            return at_tensor;
        }

        at::Tensor dropout_gen_mask_impl(const at::Tensor & self, const at::Scalar & keep_prob, const at::Scalar & seed,
        const int64_t offset, const int64_t numels)
        {
            int64_t length = (numels + 128 - 1) / 128 * 128 / 8;
            c10::TensorOptions options = self.options();
            at::Tensor mask = OpPreparation::apply_tensor_without_format(at::IntArrayRef{
                length + 32
            }, options.dtype(at::kByte));
            at::SmallVector < int64_t, ::N > offsetList = {
                0, offset
            };
            const int64_t seed1 = 0;
            OpCommand cmd;
            cmd.Name("StatelessDropOutGenMask")
            .Input(at::IntArrayRef{
                numels
            })
            .Input(keep_prob, self.scalar_type(), CompileType::MEMORY_HOST_COMPILE_DEPENDENT)
            .Input(seed, at::ScalarType::Int)
            .Input(at::Scalar(seed1), at::ScalarType::Int)
            .Input(offsetList, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
            .Output(mask)
            .Run();
            return mask;
        }

        at::Tensor dropout_gen_mask_dispatch(const at::Tensor & self, const at::Scalar & keep_prob, const at::Scalar & seed,
        const int64_t offset, const int64_t numels, const bool gen_mask_parallel, const bool sync)
        {
            at::Tensor mask;

            if (gen_mask_parallel) {
                auto original_stream = c10_npu::getCurrentNPUStream();
                {
                    // During the life cycle of this raii instance, the calcu stream is set as the
                    // secondary stream, and tasks are distributed to the secondary stream. At the
                    // same time, according to the one-stream-one-pool principle, memory is also
                    // alloced from the pool of the secondary stream.
                    c10_npu::SecondaryStreamGuard guard(c10_npu::getCurrentSecondaryStream());
                    mask = dropout_gen_mask_impl(self, keep_prob, seed, offset, numels);
                    if (sync) {
                        NPU_CHECK_ERROR (c10_npu::acl::AclrtSynchronizeStreamWithTimeout(original_stream));
                    }
                }
            } else {
                mask = dropout_gen_mask_impl(self, keep_prob, seed, offset, numels);
            }
            return mask;
        }
    }
    // namespace _

    at::Tensor dropout_gen_mask(const at::Tensor & self, double keep_prob, int64_t head_num, std::string input_layout,
    bool gen_mask_parallel, bool sync, int64_t & seed, int64_t & offset, int64_t & numels)
    {
        at::Tensor drop_mask;
        if (input_layout == "BSH") {
            numels = self.size(0) * head_num * self.size(1) * self.size(1);
            // [B,N,S,S]
        } else
            if (input_layout == "SBH") {
                numels = self.size(1) * head_num * self.size(0) * self.size(0);
                // [B,N,S,S]
            }
        int64_t length = (numels + 128 - 1) / 128 * 128 / 8;
        length += 32;
        if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_NORMAL) {
            const auto gen = at_npu::detail::getDefaultNPUGenerator();
            auto pair = at::check_generator < at_npu::NPUGeneratorImpl >(gen)->philox_engine_inputs(10);
            seed = pair.first;
            offset = pair.second;
            drop_mask = dropout_gen_mask_dispatch(self, at::Scalar(keep_prob), at::Scalar(seed), offset, numels, gen_mask_parallel, sync);
        } else
            if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_ALL) {
                drop_mask = at::zeros(at::IntArrayRef{
                    length
                }, self.options().dtype(at::kByte));
            }
        return drop_mask;
    }

    std::tuple < at::Tensor, at::Tensor, at::Tensor, at::Tensor > npu_fusion_attention_backward(const at::Tensor & query,
    const at::Tensor & key,
    const at::Tensor & value,
    const at::Tensor & dy,
    int64_t head_num,
    const std::string input_layout,
    const c10::optional < at::Tensor > & pse,
    const c10::optional < at::Tensor > & drop_mask,
    const c10::optional < at::Tensor > & padding_mask,
    const c10::optional < at::Tensor > & atten_mask,
    const c10::optional < at::Tensor > & softmax_max,
    const c10::optional < at::Tensor > & softmax_sum,
    const c10::optional < at::Tensor > & softmax_in,
    const c10::optional < at::Tensor > & attention_in,
    double scale_value,
    double keep_prob,
    int64_t pre_tockens,
    int64_t next_tockens,
    int64_t inner_precise)
    {
        double scale = scale_value;

        const at::Tensor & pse_const = pse.value_or(at::Tensor());
        const at::Tensor & drop_mask_const = drop_mask.value_or(at::Tensor());
        const at::Tensor & padding_mask_const = padding_mask.value_or(at::Tensor());
        const at::Tensor & atten_mask_const = atten_mask.value_or(at::Tensor());
        const at::Tensor & softmax_max_const = softmax_max.value_or(at::Tensor());
        const at::Tensor & softmax_sum_const = softmax_sum.value_or(at::Tensor());
        const at::Tensor & softmax_const = softmax_in.value_or(at::Tensor());
        const at::Tensor & attention_const = attention_in.value_or(at::Tensor());

        at::Tensor format_query = format_trans(query);
        at::Tensor format_key = format_trans(key);
        at::Tensor format_value = format_trans(value);
        at::Tensor format_dy = format_trans(dy);

        at::Tensor format_pse = format_trans(pse_const);
        at::Tensor format_drop_mask = format_trans(drop_mask_const);
        at::Tensor format_padding_mask = format_trans(padding_mask_const);
        at::Tensor format_atten_mask = format_trans(atten_mask_const);
        at::Tensor format_softmax_max = format_trans(softmax_max_const);
        at::Tensor format_softmax_sum = format_trans(softmax_sum_const);
        at::Tensor format_softmax = format_trans(softmax_const);
        at::Tensor format_attention = format_trans(attention_const);
        at::Tensor dq = OpPreparation::apply_tensor_without_format(format_query);
        at::Tensor dk = OpPreparation::apply_tensor_without_format(format_key);
        at::Tensor dv = OpPreparation::apply_tensor_without_format(format_value);
        char* input_layout_ptr = const_cast < char * > (input_layout.c_str());
        at::Tensor dpse;
        if (format_pse.defined()) {
            dpse = OpPreparation::apply_tensor_without_format(format_pse);
        } else {
            dpse = at::empty({
                0
            }, query.options());
        }

        EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnFlashAttentionScoreGrad, format_query, format_key, format_value, format_dy,
        format_pse, format_drop_mask, format_padding_mask, format_atten_mask,
        format_softmax_max, format_softmax_sum, format_softmax, format_attention, scale_value, keep_prob,
        pre_tockens, next_tockens, head_num, input_layout_ptr, inner_precise, dq, dk, dv, dpse);

        if (!format_pse.defined()) {
            at::Tensor dpse_required;
            dpse = dpse_required;
        }

        return std::make_tuple(dq, dk, dv, dpse);
    }

    std::tuple < at::Tensor, at::Tensor, at::Tensor, at::Tensor > npu_fusion_attention_grad(const at::Tensor & query,
    const at::Tensor & key,
    const at::Tensor & value,
    const at::Tensor & dy,
    int64_t head_num,
    c10::string_view input_layout,
    const c10::optional < at::Tensor > & pse,
    const c10::optional < at::Tensor > & padding_mask,
    const c10::optional < at::Tensor > & atten_mask,
    const c10::optional < at::Tensor > & softmax_max,
    const c10::optional < at::Tensor > & softmax_sum,
    const c10::optional < at::Tensor > & softmax_in,
    const c10::optional < at::Tensor > & attention_in,
    double scale_value,
    double keep_prob,
    int64_t pre_tockens,
    int64_t next_tockens,
    int64_t inner_precise,
    int64_t seed,
    int64_t offset,
    int64_t numels,
    bool gen_mask_parallel,
    bool sync)
    {
        TORCH_CHECK(query.dim() == 3, "The shapes of the input query should be 3-dimensional, but got ", query.dim(), "-dimensional");
        TORCH_CHECK(key.dim() == 3, "The shapes of the input key should be 3-dimensional, but got ", key.dim(), "-dimensional");
        TORCH_CHECK(value.dim() == 3, "The shapes of the input value should be 3-dimensional, but got ", value.dim(), "-dimensional");
        TORCH_CHECK(dy.dim() == 3, "The shapes of the input dy should be 3-dimensional, but got ", dy.dim(), "-dimensional");
        TORCH_CHECK(keep_prob >= 0 && keep_prob <= 1, "The keep_prob value must be in range of [0, 1], but got ", keep_prob);
        std::string input_layout_str = std::string(input_layout);
        for (auto & c : input_layout_str) {
            c = toupper(c);
        }
        TORCH_CHECK(input_layout_str == "BSH" || input_layout_str == "SBH",
        "The input_layout should be BSH/SBH(case-insensitive), but got ", input_layout);
        int64_t length = (numels + 128 - 1) / 128 * 128 / 8;
        length += 32;
        at::Tensor drop_mask;
        if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_NORMAL) {
            drop_mask = dropout_gen_mask_dispatch(query, at::Scalar(keep_prob), at::Scalar(seed), offset, numels, gen_mask_parallel, sync);
        } else
            if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_ALL) {
                drop_mask = at::zeros(at::IntArrayRef{
                    length
                }, query.options().dtype(at::kByte));
            }
        auto result = npu_fusion_attention_backward(query,
        key, value, dy, head_num, input_layout_str, pse, drop_mask, padding_mask, atten_mask,
        softmax_max, softmax_sum, softmax_in, attention_in, scale_value,
        keep_prob, pre_tockens, next_tockens, inner_precise);
        if (!sync) {
            c10_npu::NPUEvent npu_event;
            npu_event.record(c10_npu::getCurrentNPUStream());
            npu_event.block(c10_npu::getCurrentSecondaryStream());
        }

        return result;
    }

    std::tuple < at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t > npu_fusion_attention(const at::Tensor & query, const at::Tensor & key,
    const at::Tensor & value, int64_t head_num, c10::string_view input_layout,
    const c10::optional < at::Tensor > & pse_opt, const c10::optional < at::Tensor > & padding_mask_opt,
    const c10::optional < at::Tensor > & atten_mask_opt,
    double scale, double keep_prob, int64_t pre_tockens, int64_t next_tockens, int64_t inner_precise,
    bool gen_mask_parallel, bool sync)
    {
        const at::Tensor & pse = pse_opt.value_or(at::Tensor());
        const at::Tensor & padding_mask = padding_mask_opt.value_or(at::Tensor());
        const at::Tensor & atten_mask = atten_mask_opt.value_or(at::Tensor());

        TORCH_CHECK(query.dim() == 3, "The shapes of the input query should be 3-dimensional, but got ", query.dim(), "-dimensional");
        TORCH_CHECK(key.dim() == 3, "The shapes of the input key should be 3-dimensional, but got ", key.dim(), "-dimensional");
        TORCH_CHECK(value.dim() == 3, "The shapes of the input value should be 3-dimensional, but got ", value.dim(), "-dimensional");
        TORCH_CHECK(keep_prob >= 0 && keep_prob <= 1, "The keep_prob value must be in range of [0, 1], but got ", keep_prob);
        std::string input_layout_str = std::string(input_layout);
        for (auto & c : input_layout_str) {
            c = toupper(c);
        }
        TORCH_CHECK(input_layout_str == "BSH" || input_layout_str == "SBH",
        "The input_layout should be BSH/SBH(case-insensitive), but got ", input_layout);

        int64_t B = 0;
        int64_t S0 = 0;
        // S for query
        int64_t S1 = 0;
        // S for key & value
        int64_t H = 0;
        if (input_layout_str == "BSH") {
            B = query.size(0);
            S0 = query.size(1);
            S1 = key.size(1);
            H = query.size(2);
        } else
            if (input_layout_str == "SBH") {
                B = query.size(1);
                S0 = query.size(0);
                S1 = key.size(0);
                H = query.size(2);
            }

        double scale_value = scale;

        at::Tensor format_query = format_trans(query);
        at::Tensor attention_score = OpPreparation::apply_tensor_without_format(format_query);
        at::Tensor format_key = format_trans(key);
        at::Tensor format_value = format_trans(value);

        at::Tensor format_pse = format_trans(pse);
        at::Tensor format_padding_mask = format_trans(padding_mask);
        at::Tensor format_atten_mask = format_trans(atten_mask);

        int64_t seed;
        int64_t offset;
        int64_t numels;
        at::Tensor format_drop_mask = dropout_gen_mask(format_query, keep_prob, head_num, input_layout_str,
        gen_mask_parallel, sync, seed, offset, numels);

        at::Tensor softmax_max;
        at::Tensor softmax_sum;
        at::Tensor softmax_out;

        softmax_max = OpPreparation::apply_tensor_without_format({
            B, head_num, S0, 8
        },
        query.options().dtype(at::kFloat));
        // [B, N, S0, 8]
        softmax_sum = OpPreparation::apply_tensor_without_format({
            B, head_num, S0, 8
        },
        query.options().dtype(at::kFloat));
        // [B, N, S0, 8]
        softmax_out = at::empty({
            0
        }, query.options());

        char* input_layout_ptr = const_cast < char * > (input_layout_str.c_str());
        EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnFlashAttentionScore, format_query, format_key, format_value,
        format_pse, format_drop_mask, format_padding_mask, format_atten_mask,
        scale, keep_prob, pre_tockens, next_tockens, head_num, input_layout_ptr, inner_precise,
        softmax_max, softmax_sum, softmax_out, attention_score);

        if (!sync) {
            c10_npu::NPUEvent npu_event;
            npu_event.record(c10_npu::getCurrentNPUStream());
            npu_event.block(c10_npu::getCurrentSecondaryStream());
        }

        return std::make_tuple(attention_score, softmax_max, softmax_sum, softmax_out,
        seed, offset, numels);
    }
}
// namespace op_api
