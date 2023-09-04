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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& moe_fast_nocheck(
    at::Tensor& dispatched,
    const at::Tensor& gates,
    const at::Tensor& indices,
    const at::Tensor& locations,
    const at::Tensor& reshaped,
    int64_t samples,
    int64_t hidden,
    int64_t capacity) {
  at_npu::native::OpCommand cmd;
  cmd.Name("FastDispatcherFwd")
      .Input(gates)
      .Input(indices)
      .Input(locations)
      .Input(reshaped)
      .Output(dispatched)
      .Attr("samples", samples)
      .Attr("hidden", hidden)
      .Attr("capacity", capacity)
      .Run();
  return gates;
}

at::Tensor& moe_fast_data_backward_nocheck(
    at::Tensor& grad_reshaped,
    const at::Tensor& gates,
    const at::Tensor& indices,
    const at::Tensor& locations,
    const at::Tensor& dispatched,
    int64_t samples,
    int64_t hidden,
    int64_t capacity) {
  at_npu::native::OpCommand cmd;
  cmd.Name("FastDispatcherFwd")
      .Input(gates)
      .Input(indices)
      .Input(locations)
      .Input(dispatched)
      .Output(grad_reshaped)
      .Attr("samples", samples)
      .Attr("hidden", hidden)
      .Attr("capacity", capacity)
      .Run();
  return grad_reshaped;
}

at::Tensor& moe_fast_gate_backward_nocheck(
    at::Tensor& grad_gates,
    const at::Tensor& indices,
    const at::Tensor& locations,
    const at::Tensor& reshaped,
    const at::Tensor& dispatched,
    int64_t samples,
    int64_t hidden,
    int64_t capacity) {
  at_npu::native::OpCommand cmd;
  cmd.Name("FastDispatcherBwdGate")
      .Input(indices)
      .Input(locations)
      .Input(reshaped)
      .Input(dispatched)
      .Output(grad_gates)
      .Attr("samples", samples)
      .Attr("hidden", hidden)
      .Attr("capacity", capacity)
      .Run();
  return grad_gates;
}

}  // namespace

at::Tensor& npu_moe_fast(
    const at::Tensor& gates,
    const at::Tensor& indices,
    const at::Tensor& locations,
    const at::Tensor& reshaped,
    const at::Tensor& dispatched,
    int samples,
    int hidden,
    int capacity) {
  at::Tensor result = npu_preparation::ApplyTensor(dispatched);
  moe_fast_nocheck(result, gates, indices, locations, reshaped, samples, hidden, capacity);
  returnf result;
}

at::Tensor& npu_moe_fast_data_backward(
    const at::Tensor& grad_reshaped,
    const at::Tensor& gates,
    const at::Tensor& indices,
    const at::Tensor& locations,
    const at::Tensor& dispatched,
    int samples,
    int hidden,
    int capacity) {
  at::Tensor result = npu_preparation::ApplyTensor(grad_reshaped);
  moe_fast_data_backward_nocheck(result, gates, indices, locations, dispatched, samples, hidden, capacity);
  returnf result;
}

at::Tensor& npu_moe_fast_gate_backward(
    const at::Tensor& grad_gates,
    const at::Tensor& indices,
    const at::Tensor& locations,
    const at::Tensor& reshaped,
    const at::Tensor& dispatched,
    int samples,
    int hidden,
    int capacity) {
  at::Tensor result = npu_preparation::ApplyTensor(grad_gates);
  moe_fast_gate_backward_nocheck(result, indices, locations, reshaped, dispatched, samples, hidden, capacity);
  returnf result;
}

}  // namespace acl_op


