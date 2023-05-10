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
using calcu_op_util = at_npu::native::CalcuOpUtil;

std::tuple<at::Tensor, at::Tensor> _ctc_loss(
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    at::IntArrayRef input_lengths_list,
    at::IntArrayRef target_lengths_list,
    int64_t blank,
    bool zero_infinity) {
  at::Tensor log_probs_cast = log_probs;
  if (log_probs.scalar_type() == at::kHalf) {
    log_probs_cast = op_plugin::npu_dtype_cast(log_probs_cast, at::kFloat);
  }

  auto input_lengths_tensor = at::tensor(input_lengths_list, targets.options());
  auto target_lengths_tensor = at::tensor(target_lengths_list, targets.options());

  int64_t max_length = 0;
  for (auto &i : target_lengths_list) {
    if (i > max_length) {
      max_length = i;
    }
  }
  // add max_length info
  auto shape = log_probs.sizes();
  blank = blank + max_length * shape[2];

  auto output_sizes = op_infer::ctc_loss_npu_output_size(log_probs, max_length);
  at::Tensor neg_log_likelihood = npu_preparation::ApplyTensorWithFormat(
      std::get<0>(output_sizes),
      log_probs_cast.options(),
      calcu_op_util::GetTensorNpuFormat(log_probs_cast));

  at::Tensor log_alpha = npu_preparation::ApplyTensorWithFormat(
      std::get<1>(output_sizes),
      log_probs_cast.options(),
      calcu_op_util::GetTensorNpuFormat(log_probs_cast));

  at_npu::native::OpCommand cmd;
  cmd.Name("CTCLossV2")
      .Input(log_probs_cast)
      .Input(targets)
      .Input(input_lengths_tensor)
      .Input(target_lengths_tensor)
      .Output(neg_log_likelihood)
      .Output(log_alpha)
      .Attr("blank", blank)
      .Attr("zero_infinity", zero_infinity)
      .Run();

  if (log_probs.scalar_type() == at::kHalf) {
    neg_log_likelihood = op_plugin::npu_dtype_cast(neg_log_likelihood, at::kHalf);
    log_alpha = op_plugin::npu_dtype_cast(log_alpha, at::kHalf);
  }

  return std::tie(neg_log_likelihood, log_alpha);
}

at::Tensor ctc_loss(
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    at::IntArrayRef input_lengths_list,
    at::IntArrayRef target_lengths_list,
    int64_t blank,
    int64_t reduction,
    bool zero_infinity) {
  at::Tensor res = std::get<0>(at::_ctc_loss(
      log_probs,
      targets,
      input_lengths_list,
      target_lengths_list,
      blank,
      zero_infinity));

  if (zero_infinity) {
    res = at::where(res == at::Scalar(std::numeric_limits<double>::infinity()), at::zeros({}, res.options()), res);
  }

  if (reduction == at::Reduction::Mean) {
    std::vector<int64_t> target_lengths_vector = target_lengths_list.vec();
    auto target_lengths_tensor = calcu_op_util::CopyTensorHostToDevice(
        at::from_blob(target_lengths_vector.data(), {target_lengths_vector.size()}, at::kLong)).clamp_min(1);
    at::Tensor target_lengths_tensor_ = target_lengths_tensor.to(res.dtype());
    return (res / target_lengths_tensor_).mean();
  } else if (reduction == at::Reduction::Sum) {
    return res.sum();
  }

  return res;
}

at::Tensor ctc_loss(
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    const at::Tensor& input_lengths,
    const at::Tensor& target_lengths,
    int64_t blank,
    int64_t reduction,
    bool zero_infinity) {
  TORCH_CHECK(isIntegralType(input_lengths.scalar_type(), false), "input_lengths must be integral");
  TORCH_CHECK(isIntegralType(target_lengths.scalar_type(), false), "target_lengths must be integral");

  at::Tensor input_lengths_tensor = input_lengths.to(at::Device(at::kCPU), at::kLong).contiguous();
  at::Tensor target_lengths_tensor = target_lengths.to(at::Device(at::kCPU), at::kLong).contiguous();

  at::IntArrayRef input_lengths_list(input_lengths_tensor.data_ptr<int64_t>(), input_lengths_tensor.numel());
  at::IntArrayRef target_lengths_list(target_lengths_tensor.data_ptr<int64_t>(), target_lengths_tensor.numel());

  return at::ctc_loss(log_probs, targets, input_lengths_list, target_lengths_list, blank, reduction, zero_infinity);
}
} // namespace op_plugin
