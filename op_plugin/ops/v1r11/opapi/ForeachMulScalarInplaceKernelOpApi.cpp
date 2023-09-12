#include <ATen/native/ForeachUtils.h>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

void _foreach_mul_(const at::TensorList self, const at::Scalar& scalar) {
    at::Tensor scalar_tensor = at_npu::native::CalcuOpUtil::CopyScalarToDevice(scalar, self[0].scalar_type());

    EXEC_NPU_CMD(aclnnForeachMulScalarInplace, scalar_tensor);
}

}