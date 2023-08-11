#include "op_plugin/ops/OpInterface.h"
#include "op_plugin/ops/OpApiInterface.h"

namespace op_plugin {
at::Tensor bitwise_or(const at::Tensor & self, const at::Scalar & other) {
    return op_api::bitwise_or(self, other);
}

at::Tensor& bitwise_or_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result) {
    return op_api::bitwise_or_out(self, other, result);
}

at::Tensor bitwise_or(const at::Tensor& self, const at::Tensor& other) {
    return op_api::bitwise_or(self, other);
}

at::Tensor& bitwise_or_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
    return op_api::bitwise_or_out(self, other, result);
}

at::Tensor bitwise_xor(const at::Tensor& self, const at::Scalar& other) {
    return op_api::bitwise_xor(self, other);
}

at::Tensor& bitwise_xor_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result) {
    return op_api::bitwise_xor_out(self, other, result);
}

at::Tensor bitwise_xor(const at::Tensor& self, const at::Tensor& other) {
    return op_api::bitwise_xor(self, other);
}

at::Tensor& bitwise_xor_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
    return op_api::bitwise_xor_out(self, other, result);
}

at::Tensor& cumsum_out(const at::Tensor& self, int64_t dim, c10::optional<at::ScalarType> dtype, at::Tensor& result) {
    return op_api::cumsum_out(self, dim, dtype, result);
}

at::Tensor& cumsum_out(const at::Tensor& self, at::Dimname dim, c10::optional<at::ScalarType> dtype,
                       at::Tensor& result) {
    return op_api::cumsum_out(self, dim, dtype, result);
}

at::Tensor& dot_out(const at::Tensor& self, const at::Tensor& tensor, at::Tensor& result) {
    return op_api::dot_out(self, tensor, result);
}

at::Tensor dot(const at::Tensor& self, const at::Tensor& tensor) {
    return op_api::dot(self, tensor);
}

at::Tensor& erf_out(const at::Tensor& self, at::Tensor& result) {
    return op_api::erf_out(self, result);
}

at::Tensor& erf_(at::Tensor& self) {
    return op_api::erf_(self);
}

at::Tensor erf(const at::Tensor& self) {
    return op_api::erf(self);
}

at::Tensor& exp_out(const at::Tensor& self, at::Tensor& result) {
    return op_api::exp_out(self, result);
}

at::Tensor& exp_(at::Tensor& self) {
    return op_api::exp_(self);
}

at::Tensor exp(const at::Tensor& self) {
    return op_api::exp(self);
}

at::Tensor& fill_(at::Tensor& self, const at::Scalar& value) {
    return op_api::fill_(self, value);
}

at::Tensor& fill_(at::Tensor& self, const at::Tensor& other) {
    return op_api::fill_(self, other);
}

at::Tensor flip(const at::Tensor& self, at::IntArrayRef dims) {
    return op_api::flip(self, dims);
}

at::Tensor& floor_out(const at::Tensor& self, at::Tensor& result) {
    return op_api::floor_out(self, result);
}

at::Tensor& floor_(at::Tensor& self) {
    return op_api::floor_(self);
}

at::Tensor floor(const at::Tensor& self) {
    return op_api::floor(self);
}

at::Tensor& fmod_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
    return op_api::fmod_out(self, other, result);
}

at::Tensor fmod(const at::Tensor& self, const at::Tensor& other) {
    return op_api::fmod(self, other);
}

at::Tensor &fmod_out(const at::Tensor &self, const at::Scalar& other, at::Tensor &result) {
    return op_api::fmod_out(self, other, result);
}

at::Tensor fmod(const at::Tensor &self, const at::Scalar& other) {
    return op_api::fmod(self, other);
}

at::Tensor& fmod_(at::Tensor& self, const at::Tensor& other) {
    return op_api::fmod_(self, other);
}

at::Tensor& fmod_(at::Tensor& self, const at::Scalar& other) {
    return op_api::fmod_(self, other);
}

at::Tensor hardtanh_backward(const at::Tensor& grad_output, const at::Tensor& self, const at::Scalar& min_val,
                             const at::Scalar& max_val) {
    return op_api::hardtanh_backward(grad_output, self, min_val, max_val);
}

at::Tensor hardtanh(const at::Tensor& self, const at::Scalar& min, const at::Scalar& max) {
    return op_api::hardtanh(self, min, max);
}

at::Tensor& hardtanh_(at::Tensor& self, const at::Scalar& min, const at::Scalar& max) {
    return op_api::hardtanh_(self, min, max);
}

at::Tensor& inverse_out(const at::Tensor& self, at::Tensor& result) {
    return op_api::inverse_out(self, result);
}

at::Tensor inverse(const at::Tensor& self) {
    return op_api::inverse(self);
}

at::Tensor isclose(const at::Tensor& self, const at::Tensor& other, double rtol, double atol, bool equal_nan) {
    return op_api::isclose(self, other, rtol, atol, equal_nan);
}

at::Tensor& addcdiv_out(const at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2,
                        const at::Scalar& value, at::Tensor& result) {
  return op_api::addcdiv_out(self, tensor1, tensor2, value, result);
}

at::Tensor addcdiv(const at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2,
                   const at::Scalar& value) {
  return op_api::addcdiv(self, tensor1, tensor2, value);
}

at::Tensor& addcdiv_(at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2, const at::Scalar& value) {
  return op_api::addcdiv_(self, tensor1, tensor2, value);
}

at::Tensor embedding(const at::Tensor& weight, const at::Tensor& indices, int64_t padding_idx, bool scale_grad_by_freq,
                     bool sparse) {
  return op_api::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse);
}

at::Tensor embedding_dense_backward(const at::Tensor& grad_weight, const at::Tensor& indices, int64_t num_weights,
                                    int64_t padding_idx, bool scale_grad_by_freq) {
  return op_api::embedding_dense_backward(grad_weight, indices, num_weights, padding_idx, scale_grad_by_freq);
}

at::Tensor embedding_backward(const at::Tensor& grad, const at::Tensor& indices, int64_t num_weights,
                              int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
  return op_api::embedding_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse);
}

at::Tensor grid_sampler_2d(const at::Tensor& self, const at::Tensor& grid, int64_t interpolation_mode,
                           int64_t padding_mode, bool align_corners) {
  return op_api::grid_sampler_2d(self, grid, interpolation_mode, padding_mode, align_corners);
}

std::tuple<at::Tensor, at::Tensor> grid_sampler_2d_backward(const at::Tensor& grad, const at::Tensor& input,
                                                            const at::Tensor& grid, int64_t interpolation_mode,
                                                            int64_t padding_mode, bool align_corners,
                                                            std::array<bool,2> output_mask) {
  return op_api::grid_sampler_2d_backward(grad, input, grid, interpolation_mode, padding_mode, align_corners,
                                          output_mask);
}

at::Tensor& erfc_out(const at::Tensor& self, at::Tensor& result) {
  return op_api::erfc_out(self, result);
}

at::Tensor& erfc_(at::Tensor& self) {
  return op_api::erfc_(self);
}

at::Tensor erfc(const at::Tensor& self) {
  return op_api::erfc(self);
}
} // namespace op_plugin
