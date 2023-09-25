#include <ATen/native/ForeachUtils.h>
#include <op_plugin/OpApiInterface.h>
#include <op_plugin/utils/op_api_common.h>

namespace op_api {
		using npu_preparation = at_npu::native::OpPreparation;

		void _foreach_add_(at::TensorList self, at::TensorList other, const at::Scalar& scalar)
		{
				print("_foreach_add_list_inplace begin \n");
				at::Tensor scalar_tensor = at_npu::native::CalcuOpUtil::CopyScalarToDevice(scalar, self[0].scalar_type());
				print("_foreach_add_list_inplace end \n");
				EXEC_NPU_CMD(aclnnForeachAddListInplace, self, other, scalar_tensor);
				}

}