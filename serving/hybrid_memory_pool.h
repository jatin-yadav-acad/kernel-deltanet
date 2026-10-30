// serving/hybrid_memory_pool.h
#pragma once
#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#else
// Standalone Host Fallback for C++ Compilation & Non-GPU Host Testing
#include <cstdlib>
#include <cstring>
typedef void* cudaStream_t;
typedef int cudaError_t;
#define cudaSuccess 0
#define cudaMemcpyDeviceToDevice 0
inline cudaError_t cudaMalloc(void** ptr, size_t size) { *ptr = std::malloc(size); return cudaSuccess; }
inline cudaError_t cudaFree(void* ptr) { std::free(ptr); return cudaSuccess; }
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, int kind, cudaStream_t stream = nullptr) {
    std::memcpy(dst, src, count);
    return cudaSuccess;
}
inline const char* cudaGetErrorString(cudaError_t) { return "success"; }
#endif
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <iostream>

/**
 * Production Hybrid Memory Pool for Kernel-DeltaNet (20:2 Topology).
 * 
 * Hardware-Memory Allocation Policy:
 * 1. Recurrent Layers (20 layers): Pre-allocates a contiguous flat memory pool.
 *    Each stream occupies exactly M_state = 22.0 MB fixed across T=1..65536.
 *    Zero DRAM allocation or fragmentation during autoregressive decoding.
 * 2. Boundary Layers (2 Softmax Anchor layers): Allocates 16-token page-slabs in FP8.
 *    Slashes KV cache allocation by 90.9% to 96.5% vs monolithic Softmax models.
 */

struct StreamSlot {
    int stream_id;
    int prompt_len;
    int generated_tokens;
    void* d_recurrent_buffer; // Pointer to contiguous 22 MB state
    std::vector<int> anchor_page_indices; // 16-token page IDs for the 2 Softmax layers
    bool is_active;
};

class HybridMemoryPool {
public:
    HybridMemoryPool(size_t max_streams, size_t max_seq_len, cudaStream_t stream = nullptr);
    ~HybridMemoryPool();

    // Allocate / Register a new serving stream
    int allocate_stream(int prompt_len);

    // Fork stream (zero-copy prefix forking via async L2/DRAM copy)
    int fork_stream(int parent_stream_id);

    // Release stream upon completion
    void release_stream(int stream_id);

    // Memory Accounting & Telemetry
    size_t get_total_allocated_bytes() const;
    size_t get_free_bytes() const;
    void print_memory_profile() const;

private:
    size_t max_streams_;
    size_t max_seq_len_;
    cudaStream_t stream_;

    // Recurrent Slab: max_streams x 22.0 MB
    void* d_recurrent_slab_base_;
    size_t recurrent_bytes_per_stream_;
    size_t total_recurrent_pool_bytes_;

    // Softmax Paged Pool: Paged blocks of 16 tokens in FP8 for 2 layers
    void* d_paged_kv_pool_base_;
    size_t page_size_bytes_;
    size_t num_total_pages_;
    std::vector<int> free_page_stack_;
    std::vector<int> page_ref_counts_;

    // Stream Tracking Table
    std::vector<StreamSlot> stream_slots_;
    std::vector<int> free_stream_slots_;
};
