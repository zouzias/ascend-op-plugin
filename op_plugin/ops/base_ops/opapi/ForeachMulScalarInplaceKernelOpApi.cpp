#include <ATen/native/ForeachUtils.h>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

void _foreach_mul_(const at::TensorList self, const at::Scalar& scalar) {
    auto self_size = self.size()
    int empty_size = 0;

    if (self.empty()) {
        return;
    }
    for (int i = 0; i < self_size; i++) {
        empty_size += 1;
    }
    if (empty_size == self_size) {
        return;
    }
    auto scalar_type = self[0].scalar_type();
    bool is_support = true;
    if (scalar_type != at::ScalarType::Half && scalar_type != at::ScalarType::Float && scalar_type != at::ScalarType::Int) {
        is_support = false;
        TORCH_CHECK(is_support, "input must be half, float or int32");
    }
    at::Tensor scalar_tensor = at_npu::native::CalcuOpUtil::CopyScalarToDevice(scalar, self[0].scalar_type());
    EXEC_NPU_CMD(aclnnForeachMulScalarInplace, scalar_tensor);
}
}