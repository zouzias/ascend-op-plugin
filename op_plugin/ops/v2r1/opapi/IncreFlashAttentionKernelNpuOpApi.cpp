#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_incre_flash_attention(
    const at::Tensor &query, const at::Tensor &key,
    const at::Tensor &value, int64_t head_num, c10::string_view input_layout, double scale,
    const c10::optional<at::Tensor> &padding_mask,
    const c10::optional<at::Tensor> &atten_mask,
    c10::OptionalIntArrayRef actual_seq_lengths,
    int64_t kv_head_num)
{
  // construct the output tensor of the NPU
  auto output = npu_preparation::apply_tensor_without_format(query);

  // convert str
  std::string input_layout_str = std::string(input_layout);
  char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());

  at::TensorList keyTensors = {key};
  at::TensorList valueTensors = {value};

  auto actSeqLen = (actual_seq_lengths.has_value()) ? actual_seq_lengths.value().vec() : std::vector<at::IntArrayRef::value_type>{};

  // dispatch hostAPI
  EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnIncreFlashAttention, query, key, value, padding_mask, atten_mask, s,
                               head_num, scale, input_layout_ptr, kv_head_num, output);
  return output;
}
} // namespace op_api