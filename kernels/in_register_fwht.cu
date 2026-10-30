// kernels/in_register_fwht.cu
#include "in_register_fwht.cuh"
#include <torch/extension.h>

/**
 * CUDA Global Kernel: in_register_fwht_dual_lifting_kernel
 * Applies in-register Fast Walsh-Hadamard Transform butterfly network directly
 * within GPU Warp Registers, completely in-register before projecting onto S^{D-1}.
 */
__global__ void __launch_bounds__(512, 1) in_register_fwht_dual_lifting_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const uint64_t* __restrict__ rademacher_bits, // [M, 3] bitmasks
    int batch_size,
    int seq_len,
    int d_model,
    int D_lifted,
    float tau_star_inv
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int total_warps = batch_size * seq_len;

    if (warp_id >= total_warps) return;

    // Load 2 ambient input elements per thread (32 threads x 2 = 64 ambient dimension)
    float x[2] = {0.0f, 0.0f};
    const float* token_in = input + warp_id * d_model;
    if (2 * lane_id < d_model) {
        x[0] = token_in[2 * lane_id];
    }
    if (2 * lane_id + 1 < d_model) {
        x[1] = token_in[2 * lane_id + 1];
    }

    // Unpack rademacher bits into local constant registers
    uint64_t local_masks[4][3];
    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        #pragma unroll
        for (int s = 0; s < 3; ++s) {
            local_masks[m][s] = rademacher_bits ? rademacher_bits[m * 3 + s] : 0xAAAAAAAAAAAAAAAAULL;
        }
    }

    // Execute in-register FWHT dual lifting
    float rff_out[16];
    in_register_fwht_dual_lifting<4>(x, rff_out, lane_id, tau_star_inv, local_masks);

    // Store lifted features into global output buffer
    float* token_out = output + warp_id * D_lifted;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int out_idx = lane_id * 16 + i;
        if (out_idx < D_lifted) {
            token_out[out_idx] = rff_out[i];
        }
    }
}

torch::Tensor in_register_fwht_forward_cuda(
    torch::Tensor input,
    torch::Tensor rademacher_bits,
    int D_lifted,
    float tau_star_inv
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    int B = input.size(0);
    int T = input.size(1);
    int d_model = input.size(2);
    int total_warps = B * T;

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::empty({B, T, D_lifted}, options);

    int threads_per_block = 512;
    int total_threads = total_warps * 32;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    const uint64_t* mask_ptr = rademacher_bits.defined() ? (const uint64_t*)rademacher_bits.data_ptr() : nullptr;

    in_register_fwht_dual_lifting_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        mask_ptr,
        B, T, d_model, D_lifted, tau_star_inv
    );

    return output;
}
