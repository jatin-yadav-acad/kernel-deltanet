// kernels/in_register_fwht.cuh
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

/**
 * In-Register Fast Walsh-Hadamard Transform (FWHT) for Warp Register Files.
 * Implements vectorized intra-warp butterfly exchanges via __shfl_xor_sync across warp threads.
 * Target: Warp size 32, d_k=64, D=512 (M=4 orthogonal blocks, 16 features per thread).
 */

template <int M = 4> // M = D / (2 * d_k) = 512 / 128 = 4 orthogonal projection blocks
__device__ __forceinline__ void in_register_fwht_dual_lifting(
    const float x[2],              // 2 ambient inputs per thread (32 threads x 2 = 64)
    float rff_out[16],             // 16 lifted outputs per thread (8 cos + 8 sin; 32 x 16 = 512)
    const int lane_id,
    const float tau_star_inv,      // 1.0f / (d_k)^{0.25} = 0.35355f
    const uint64_t rademacher_bits[M][3] // Compact 64-bit diagonal sign bitmasks
) {
    float angles[8]; // 4 blocks * 2 coordinates per thread = 8 projection angles

    // 0. Pre-normalize ambient input x onto unit hypersphere S^{d_k-1}
    float x_sq = x[0] * x[0] + x[1] * x[1];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x_sq += __shfl_xor_sync(0xffffffff, x_sq, offset);
    }
    float x_norm_inv = rsqrtf(x_sq + 1e-8f);
    float x_unit[2] = { x[0] * x_norm_inv, x[1] * x_norm_inv };

    // 1. Compute 4 Structured Hadamard Blocks in Registers
    #pragma unroll
    for (int m = 0; m < M; ++m) {
        float v0 = x_unit[0];
        float v1 = x_unit[1];

        // Cascade of 3 Hadamard-Rademacher stages: H * D3 * H * D2 * H * D1
        #pragma unroll
        for (int stage = 0; stage < 3; ++stage) {
            // Apply diagonal Rademacher signs (+1 / -1) from 64-bit bitmask
            uint64_t mask = rademacher_bits[m][stage];
            v0 = ((mask >> (2 * lane_id)) & 1ULL) ? -v0 : v0;
            v1 = ((mask >> (2 * lane_id + 1)) & 1ULL) ? -v1 : v1;

            // Intra-thread radix-2 butterfly exchange
            float sum  = v0 + v1;
            float diff = v0 - v1;
            v0 = sum;
            v1 = diff;

            // 5-Round Intra-Warp Butterfly Exchange via __shfl_xor_sync across 32 threads
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                float p0 = __shfl_xor_sync(0xffffffff, v0, offset);
                float p1 = __shfl_xor_sync(0xffffffff, v1, offset);
                v0 = (lane_id & offset) ? (p0 - v0) : (v0 + p0);
                v1 = (lane_id & offset) ? (p1 - v1) : (v1 + p1);
            }
        }
        // Stage 4: Random Permutation Pi_3 via intra-warp shuffle (Eq. 3)
        // Permutes 64 coordinates across lanes via coprime stride gcd(17, 32)=1 and swaps intra-thread pair
        const int perm_lane = (lane_id * 17 + 5) & 31;
        float p_v0 = __shfl_sync(0xffffffff, (lane_id & 1) ? v1 : v0, perm_lane);
        float p_v1 = __shfl_sync(0xffffffff, (lane_id & 1) ? v0 : v1, perm_lane);
        angles[2 * m]     = p_v0;
        angles[2 * m + 1] = p_v1;
    }

    // 2. Analytical Per-Block Normalization (Folded into Compile-Time Scale)
    constexpr float FWHT_NORM = 1.0f / 512.0f;
    const float eff_scale = FWHT_NORM * tau_star_inv;

    // 3. In-Flight Dual [cos, sin] Spherical Mapping to S^{D-1} Features
    constexpr float RFF_AMP = 0.0625f; // sqrt(2.0f / D) = sqrt(2.0f / 512.0f) = 1.0f / 16.0f
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float angle = angles[i] * eff_scale;
        float s, c;
        sincosf(angle, &s, &c);
        rff_out[i]     = RFF_AMP * c;  // 8 cosine features
        rff_out[i + 8] = RFF_AMP * s;  // 8 sine features
    }
}
