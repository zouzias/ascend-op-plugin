#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
at::Tensor& log_softmax_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t dim) {
  c10::SmallVector<int64_t, N> dim_list = {dim};
  at_npu::native::OpCommand cmd;
  cmd.Name("LogSoftmaxV2")
      .Input(self)
      .Attr("axes", dim_list)
      .Output(result)
      .Run();
  return result;
}

at::Tensor log_softmax_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t dim,
    c10::optional<c10::ScalarType> dtype) {
  c10::ScalarType dst_type;
  if (dtype.has_value()) {
    dst_type = dtype.value();
  } else if (result.defined()) {
    dst_type = result.scalar_type();
  } else {
    dst_type = self.scalar_type();
  }
  // dtype same
  if (dst_type == self.scalar_type()) {
    log_softmax_nocheck(result, self, dim);
    return result;
  }

  log_softmax_nocheck(result, self.toType(dst_type), dim);
  return result;
}
} // namespace acl_op
