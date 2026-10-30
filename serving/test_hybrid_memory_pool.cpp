// serving/test_hybrid_memory_pool.cpp
#include "hybrid_memory_pool.h"
#include <iostream>
#include <chrono>
#include <cassert>
#include <vector>

int main() {
    std::cout << "==================================================================" << std::endl;
    std::cout << "  Kernel-DeltaNet Native C++ Hybrid Memory Pool Benchmark         " << std::endl;
    std::cout << "==================================================================" << std::endl;

    const size_t max_streams = 128;
    const size_t max_seq_len = 16384;

    std::cout << "[Init] Allocating slab pool for " << max_streams << " streams (max seq " << max_seq_len << ")..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    HybridMemoryPool pool(max_streams, max_seq_len);
    auto t1 = std::chrono::high_resolution_clock::now();
    double init_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[Init] Pool created in " << init_ms << " ms" << std::endl;

    // Test 1: Allocation throughput
    const int num_allocs = 100;
    std::vector<int> stream_ids;
    stream_ids.reserve(num_allocs);

    auto start_alloc = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_allocs; ++i) {
        int sid = pool.allocate_stream(2048);
        assert(sid >= 0);
        stream_ids.push_back(sid);
    }
    auto end_alloc = std::chrono::high_resolution_clock::now();
    double alloc_total_us = std::chrono::duration<double, std::micro>(end_alloc - start_alloc).count();
    double alloc_per_op_ns = (alloc_total_us / num_allocs) * 1000.0;

    std::cout << "\n[Test 1] Allocation Throughput:" << std::endl;
    std::cout << "  Allocated " << num_allocs << " streams in " << alloc_total_us << " us (" 
              << alloc_per_op_ns << " ns/op)" << std::endl;

    // Test 2: Prefix Forking Latency
    const int num_forks = 20;
    std::vector<int> child_ids;
    child_ids.reserve(num_forks);

    auto start_fork = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_forks; ++i) {
        int cid = pool.fork_stream(stream_ids[i]);
        assert(cid >= 0);
        child_ids.push_back(cid);
    }
    auto end_fork = std::chrono::high_resolution_clock::now();
    double fork_total_us = std::chrono::duration<double, std::micro>(end_fork - start_fork).count();
    double fork_per_op_us = fork_total_us / num_forks;

    std::cout << "\n[Test 2] State Prefix Forking:" << std::endl;
    std::cout << "  Forked " << num_forks << " recurrent state streams in " << fork_total_us << " us ("
              << fork_per_op_us << " us/op)" << std::endl;

    // Test 3: Print Memory Profile
    std::cout << "\n[Test 3] Memory Accounting Telemetry:" << std::endl;
    pool.print_memory_profile();

    // Test 4: Release & Deallocation
    auto start_free = std::chrono::high_resolution_clock::now();
    for (int cid : child_ids) pool.release_stream(cid);
    for (int sid : stream_ids) pool.release_stream(sid);
    auto end_free = std::chrono::high_resolution_clock::now();
    double free_total_us = std::chrono::duration<double, std::micro>(end_free - start_free).count();

    std::cout << "\n[Test 4] Release & Zero-Fragmentation Deallocation:" << std::endl;
    std::cout << "  Released all streams in " << free_total_us << " us" << std::endl;
    std::cout << "  Allocated Bytes remaining: " << pool.get_total_allocated_bytes() << " bytes (100% reclamation)" << std::endl;
    assert(pool.get_total_allocated_bytes() == 0);

    std::cout << "\n==================================================================" << std::endl;
    std::cout << "  [SUCCESS] All Native C++ Hybrid Memory Pool Tests Passed!       " << std::endl;
    std::cout << "==================================================================" << std::endl;
    return 0;
}
