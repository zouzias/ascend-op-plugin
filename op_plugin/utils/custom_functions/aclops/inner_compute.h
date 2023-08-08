at::Tensor& cal_var_out(const at::Tensor& self, at::IntArrayRef dim, const int64_t correction, const bool unbiased, const bool keepdim, at::Tensor& result);
at::Tensor cal_var(const at::Tensor& self, at::IntArrayRef dim, const int64_t correction, const bool unbiased, const bool keepdim);
std::tuple<at::Tensor, at::Tensor> cal_var_mean(const at::Tensor& self, at::IntArrayRef dim, bool unbiased, int64_t correction, bool keepdim);