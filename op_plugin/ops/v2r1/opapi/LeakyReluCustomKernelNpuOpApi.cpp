#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
at::Tensor& leaky_relu_custom_out(const at::Tensor& self, double negval,
                           at::Tensor& result) {
  at_npu::native::OpPreparation::check_tensor({self}, result, self.scalar_type(), self.sizes());
  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnLeakyReluCustom, self, negval, result);

  return result;
}

at::Tensor leaky_relu_custom(const at::Tensor& self, double negval) {
  at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(self);
  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnLeakyReluCustom, self, negval, result);

  return result;
}

} // namespace op_api
