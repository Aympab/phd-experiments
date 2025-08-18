#include "utils.hpp"
#include <random>
#include <vector>
#include <numeric>
#include <algorithm>

static inline int coprime_or_next(int stride, int n1) {
    if (std::gcd(stride, n1) == 1) return stride;
    for (int s = stride+1; s < stride + 64; ++s) {
        if (std::gcd(s, n1) == 1) return s;
    }
    return 1; // fallback (contiguous)
}

enum class AccessPattern { Contiguous, Strided, Indirect };

static std::vector<int> make_index_map(AccessPattern pat, int n1, int n2, uint64_t seed=42) {
    std::vector<int> idx(n1);
    if (pat == AccessPattern::Contiguous) {
        std::iota(idx.begin(), idx.end(), 0);
    } else if (pat == AccessPattern::Strided) {
        int stride = coprime_or_next(std::max(1, n2 % n1), n1); // tie stride to n2
        for (int i=0;i<n1;++i) idx[i] = (i * stride) % n1;
    } else { // Indirect
        std::iota(idx.begin(), idx.end(), 0);
        std::mt19937_64 rng(seed);
        std::shuffle(idx.begin(), idx.end(), rng);
    }
    return idx;
}

template <AccessPattern PAT>
static void BM_GlobalMem_SweepJ1(benchmark::State &state) {
    const auto data_range = get_range_with_constraint(state.range(0));
    const auto& n0 = data_range.get(0);
    const auto& n1 = data_range.get(1);
    const auto& n2 = data_range.get(2);

    auto Q = createSyclQueue(true, state);
    span3d_t data  (sycl_alloc(n0*n1*n2, Q), n0, n1, n2);
    span3d_t scratch(sycl_alloc(n0*n1*n2, Q), n0, n1, n2);

    Q.parallel_for(data_range, [=](auto itm){
        const int i0 = itm[0], i1 = itm[1], i2 = itm[2];
        data(i0,i1,i2)   = sycl::cos(static_cast<float>(i0 + i1 + i2));
        scratch(i0,i1,i2)= 0;
    }).wait();

    // Build index map on host, copy to device
    auto idxHost = make_index_map(PAT, n1, n2);
    int* idxDev = (int*) sycl::malloc_device(sizeof(int)*n1, Q);
    Q.memcpy(idxDev, idxHost.data(), sizeof(int)*n1).wait();

    // WG spans a full line along i1
    const int w0 = 1, w1 = n1, w2 = 1;
    sycl::range<3> local_range(w0,w1,w2);
    sycl::nd_range ndr(data_range, local_range);

    for (auto _ : state) {
        try {
            Q.submit([&](sycl::handler &cgh){
                cgh.parallel_for(ndr, [=](sycl::nd_item<3> it){
                    const int i0  = it.get_global_id(0);
                    const int i1  = it.get_global_id(1);
                    const int i2  = it.get_global_id(2);
                    const int lid = it.get_local_id(1);
                    const int lsz = it.get_local_range(1);

                    // Each lane processes a disjoint subset of j1
                    for (int t = lid; t < n1; t += lsz) {
                        const int j1 = (PAT==AccessPattern::Contiguous) ? t : idxDev[t];
                        float v = data(i0, j1, i2);
                        scratch(i0, j1, i2) = v + 0.0001f * static_cast<float>(j1);
                    }
                });
            }).wait();
        } catch (const std::exception &e) {
            state.SkipWithError(e.what());
            break;
        }
    }

    const auto n_iter = state.iterations();
    state.SetItemsProcessed(n_iter * n0 * n1 * n2);
    state.SetBytesProcessed(n_iter * n0 * n1 * n2 * sizeof(real_t) * 2);
    state.counters.insert({{"gpu", true},{"n0", n0},{"n1", n1},{"n2", n2},{"w0", w0},{"w1", w1},{"w2", w2}});

    sycl::free(idxDev, Q);
    sycl::free(data.data_handle(), Q);
    sycl::free(scratch.data_handle(), Q);
    Q.wait();
}


template <AccessPattern PAT>
static void BM_LocalMem_SweepJ1(benchmark::State &state) {
    const auto data_range = get_range_with_constraint(state.range(0));
    const auto& n0 = data_range.get(0);
    const auto& n1 = data_range.get(1);
    const auto& n2 = data_range.get(2);

    auto Q = createSyclQueue(true, state);
    span3d_t data(sycl_alloc(n0*n1*n2, Q), n0, n1, n2);

    Q.parallel_for(data_range, [=](auto itm){
        const int i0 = itm[0], i1 = itm[1], i2 = itm[2];
        data(i0,i1,i2) = sycl::cos(static_cast<float>(i0 + i1 + i2));
    }).wait();

    auto idxHost = make_index_map(PAT, n1, n2);
    int* idxDev = (int*) sycl::malloc_device(sizeof(int)*n1, Q);
    Q.memcpy(idxDev, idxHost.data(), sizeof(int)*n1).wait();

    const int w0 = 1, w1 = n1, w2 = 1;
    sycl::range<3> local_range(w0,w1,w2);
    sycl::nd_range ndr(data_range, local_range);

    for (auto _ : state) {
        try {
            Q.submit([&](sycl::handler &cgh){
                local_acc_1d local_acc(w1, cgh); // local scratch line
                cgh.parallel_for(ndr, [=](sycl::nd_item<3> it){
                    span1d_t scratch(local_acc.GET_POINTER(), w1);

                    const int i0  = it.get_global_id(0);
                    const int i1  = it.get_global_id(1);
                    const int i2  = it.get_global_id(2);
                    const int lid = it.get_local_id(1);
                    const int lsz = it.get_local_range(1);

                    // Stage line [i0,:,i2] into local
                    scratch(i1) = data(i0, i1, i2);
                    it.barrier(sycl::access::fence_space::local_space);

                    // Each lane updates a disjoint subset in-local
                    for (int t = lid; t < n1; t += lsz) {
                        const int j1 = (PAT==AccessPattern::Contiguous) ? t : idxDev[t];
                        scratch(j1) = scratch(j1) + 0.0001f * static_cast<float>(j1);
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // One cheap write-back to prevent DCE (keeps cost low)
                    if (i0 == 0 && i1 == 0 && i2 == n2-1)
                        ((span3d_t&)data)(0,0,0) = scratch(0);
                });
            }).wait();
        } catch (const std::exception &e) {
            state.SkipWithError(e.what());
            break;
        }
    }

    const auto n_iter = state.iterations();
    state.SetItemsProcessed(n_iter * n0 * n1 * n2);
    state.SetBytesProcessed(n_iter * n0 * n1 * n2 * sizeof(real_t) * 2);
    state.counters.insert({{"gpu", true},{"n0", n0},{"n1", n1},{"n2", n2},{"w0", w0},{"w1", w1},{"w2", w2}});

    sycl::free(idxDev, Q);
    sycl::free(data.data_handle(), Q);
    Q.wait();
}


// Global
static void BM_Global_Contig(benchmark::State& s){ BM_GlobalMem_SweepJ1<AccessPattern::Contiguous>(s); }
static void BM_Global_Stride(benchmark::State& s){ BM_GlobalMem_SweepJ1<AccessPattern::Strided>(s); }
static void BM_Global_Indirect(benchmark::State& s){ BM_GlobalMem_SweepJ1<AccessPattern::Indirect>(s); }

// Local
static void BM_Local_Contig(benchmark::State& s){ BM_LocalMem_SweepJ1<AccessPattern::Contiguous>(s); }
static void BM_Local_Stride(benchmark::State& s){ BM_LocalMem_SweepJ1<AccessPattern::Strided>(s); }
static void BM_Local_Indirect(benchmark::State& s){ BM_LocalMem_SweepJ1<AccessPattern::Indirect>(s); }

BENCHMARK(BM_Global_Contig   )->Name("GlobalMem_Contiguous_SweepJ1")->RangeMultiplier(2)->Range(1,1024)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Global_Stride   )->Name("GlobalMem_Stride_SweepJ1"   )->RangeMultiplier(2)->Range(1,1024)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Global_Indirect )->Name("GlobalMem_Indirect_SweepJ1" )->RangeMultiplier(2)->Range(1,1024)->UseRealTime()->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Local_Contig    )->Name("LocalMem_Contiguous_SweepJ1")->RangeMultiplier(2)->Range(1,1024)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Local_Stride    )->Name("LocalMem_Stride_SweepJ1"    )->RangeMultiplier(2)->Range(1,1024)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Local_Indirect  )->Name("LocalMem_Indirect_SweepJ1"  )->RangeMultiplier(2)->Range(1,1024)->UseRealTime()->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
