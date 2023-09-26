#include <ATen/native/ForeachUtils.h>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::vector<at::Tensor> _foreach_addcdiv(const at::TensorList input,
    const at::TensorList tensors1,
    const at::TensorList tensors2,
    const at::Tensor& scalars)
{
    auto scalars_ = at::native::convert_tensor_to_scalar_list(scalars, input.size());
    at::native::check_foreach_api_restrictions(input, tensors1, tensors2, scalars_);
    if (!at::native::can_use_fast_route({input, tensors1, tensors2}) ||
        at::native::has_integral_tensor(input, true)) {
            return at::native::foreach_tensor_addcdiv_scalarlist_slow(input, tensors1, tensors2, scalars_);
    }

    auto scalar_type = input[0].scalar_type();
    if (scalar_type != at::ScalarType::Half && scalar_type != at::ScalarType::Float) {
        TORCH_CHECK(false, "input must be half, float");
    }
    std::vector<at::Tensor> result;
    result.reserve(input.size());
    for (const at::Tensor &tensor : input) {
        auto output_size = op_infer::input_same_output_size(tensor);
        result.push_back(npu_preparation::apply_tensor_without_format(output_size, tensor.options().dtype(scalar_type)));
    }
    at::TensorList result_ = at::TensorList(result);
    auto scalar_tensor = npu_preparation::copy_tensor_host_to_device(scalars);
    EXEC_NPU_CMD(aclnnForeachAddcdivScalarList, input, tensors1, tensors2, scalar_tensor, result_);
    
    return result;
}

void _foreach_addcdiv_(const at::TensorList input,
    const at::TensorList tensors1,
    const at::TensorList tensors2,
    const at::Tensor& scalars)
{
    auto scalars_ = at::native::convert_tensor_to_scalar_list(scalars, input.size());
    at::native::check_foreach_api_restrictions(input, tensors1, tensors2, scalars_);
    if (!at::native::can_use_fast_route({input, tensors1, tensors2}) ||
        at::native::has_integral_tensor(input, true)) {
            return at::native::foreach_tensor_addcdiv_scalarlist_slow_(input, tensors1, tensors2, scalars_);
    }

    at::native::check_foreach_api_restrictions(input, tensors1, tensors2);
    auto scalar_type = input[0].scalar_type();
    if (scalar_type != at::ScalarType::Half && scalar_type != at::ScalarType::Float) {
        TORCH_CHECK(false, "input must be half, float");
    }
    auto scalar_tensor = npu_preparation::copy_tensor_host_to_device(scalars);
    EXEC_NPU_CMD(aclnnForeachAddcdivScalarList, input, tensors1, tensors2, scalar_tensor, input);
}
}
