

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> linalg_svd_out_common(const at::Tensor& A, const bool full_matrices, const bool compute_uv, at::Tensor& U, at::Tensor& S, at::Tensor& Vh);
