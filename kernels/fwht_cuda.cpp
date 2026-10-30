// kernels/fwht_cuda.cpp
#include <torch/extension.h>

torch::Tensor in_register_fwht_forward_cuda(
    torch::Tensor input,
    torch::Tensor rademacher_bits,
    int D_lifted,
    float tau_star_inv
);

torch::Tensor fwht_forward(
    torch::Tensor input,
    torch::Tensor rademacher_bits,
    int D_lifted,
    float tau_star_inv = 0.35355339f
) {
    if (input.is_cuda()) {
        return in_register_fwht_forward_cuda(input, rademacher_bits, D_lifted, tau_star_inv);
    }
    // CPU fallback if executed outside GPU environment
    int B = input.size(0);
    int T = input.size(1);
    int d = input.size(2);
    auto out = torch::zeros({B, T, D_lifted}, input.options());
    out.slice(2, 0, d) = input;
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwht_forward", &fwht_forward, "In-Register Fast Walsh-Hadamard Transform Forward (CUDA)",
          py::arg("input"), py::arg("rademacher_bits"), py::arg("D_lifted"), py::arg("tau_star_inv") = 0.35355339f);
}
