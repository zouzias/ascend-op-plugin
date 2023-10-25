#include <ATen/native/ForeachUtils.h>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

void _foreach_mul_(const at::TensorList self, const at::Scalar& scalar) {
    at::native::check_foreach_api_restrictions(self);
    if (!at::native::can_use_fast_route(self, scalar, false)) {
        return at::native::foreach_tensor_mul_scalar_kernel_slow_(self, scalar);
    }
    auto scalar_type = self[0].scalar_type();
    at::Tensor scalar_tensor = npu_preparation::copy_scalar_to_device(scalar, self[0].scalar_type());
    EXEC_NPU_CMD(aclnnForeachMulScalarInplace, self, scalar_tensor);
}
}

