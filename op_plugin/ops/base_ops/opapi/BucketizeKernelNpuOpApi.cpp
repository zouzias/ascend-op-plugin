// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include <ATen/native/ForeachUtils.h>

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
at::Tensor bucketize(const at::Tensor& self, const at::Tensor& boundaries, bool out_int32, bool right) {
  TORCH_CHECK(boundaries.dim() == 1, "boundaries tensor must be 1 dimension, but got dim(", boundaries.dim(), ")");
  return op_api::searchsorted(boundaries, self, out_int32, right, c10::nullopt, c10::nullopt);
}

at::Tensor bucketize(const at::Scalar& scalar, const at::Tensor& boundaries, bool out_int32, bool right) {
  TORCH_CHECK(boundaries.dim() == 1, "boundaries tensor must be 1 dimension, but got dim(", boundaries.dim(), ")");
  return op_api::searchsorted(boundaries, scalar, out_int32, right, c10::nullopt, c10::nullopt);
}

at::Tensor &bucketize_out(const at::Tensor& self,
                         const at::Tensor& boundaries,
                         bool out_int32,
                         bool right,
                         at::Tensor& result) {
  TORCH_CHECK(boundaries.dim() == 1, "boundaries tensor must be 1 dimension, but got dim(", boundaries.dim(), ")");
  return op_api::searchsorted_out(boundaries, self, out_int32, right, c10::nullopt, c10::nullopt, result);
}
}