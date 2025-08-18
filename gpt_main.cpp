#include "utils.hpp"
#include <random>
#include <vector>
#include <numeric>
#include <algorithm>

/* --------------------------------------
   Helpers
---------------------------------------*/
enum class AccessPattern { Contiguous, Strided, Indirect };

static inline int choose_coprime_stride(int n1) {
    // pick a small odd stride co-prime to n1 (fallback to 1 if needed)
    for (int s : {7, 11, 13, 17, 19, 23}) {
        if (std::gcd(s, n1) == 1) return s;
    }
    return 1;
}

static std::vector<int> make_index_map(AccessPattern pat, int n1, int stride, uint64_t seed=42) {
    std::vector<int> idx(n1);
    if (pat == AccessPattern::Contiguous) {
        std::iota(idx.begin(), idx.end(), 0);
    } else if (pat == AccessPattern::Strided) {
        for (int i=0;i<n1;++i) idx[i] = (i * stride) % n1;
    } else {
        std::iota(idx.begin(), idx.end(), 0);
        std::mt19937_64 rng(seed);
        std::shuffle(idx.begin(), idx.end(), rng);
    }
    return idx;
}

/* --------------------------------------
   Core kernels
   - Global memory version: read/update/write in global
   - Local memory version: stage per-WG slice in local then update
   Access along i1-dimension varies by pattern.
---------------------------------------*/
template <AccessPattern PAT>
static void BM_GlobalMem(benchmark::State &state) {
    const auto data_range = get_range_with_constraint(state.range(0));
    const auto& n0 = data_range.get(0);
    const auto& n1 = data_range.get(1);
    const auto& n2 = data_range.get(2);

    /* SYCL setup */
    auto Q = createSyclQueue(true, state);
    span3d_t data  (sycl_alloc(n0*n1*n2, Q), n0, n1, n2);
    span3d_t scratch(sycl_alloc(n0*n1*n2, Q), n0, n1, n2);

    // init
    Q.parallel_for(data_range, [=](auto itm){
        const auto& i0 = itm[0]; const auto& i1 = itm[1]; const auto& i2 = itm[2];
        data(i0,i1,i2)   = sycl::cos(static_cast<float>(i0 + i1 + i2));
        scratch(i0,i1,i2)= 0;
    }).wait();

    // index map for i1 dimension (host) -> device USM
    const int stride = choose_coprime_stride(n1);
    auto idxHost = make_index_map(PAT, n1, stride);
    int* idxDev = (int*) sycl::malloc_device(sizeof(int)*n1, Q);
    Q.memcpy(idxDev, idxHost.data(), sizeof(int)*n1).wait();

    // choose local size (1, n1, 1) to keep per-WG line along i1
    const size_t w0 = 1, w1 = n1, w2 = 1;
    sycl::range<3> local_range(w0,w1,w2);
    sycl::nd_range ndr(data_range, local_range);

    if(stride!=1){
        for(int i=0;i<10;++i){
            std::cout << idxHost[i] << std::endl;
        }
    }

    /* Benchmark */
    for (auto _ : state) {
        try {
            Q.submit([&](sycl::handler &cgh){
                cgh.parallel_for(ndr, [=](auto itm){
                    const int i0 = itm.get_global_id(0);
                    const int i1 = itm.get_global_id(1);
                    const int i2 = itm.get_global_id(2);

                    // map i1 -> j1 according to pattern
                    int j1;
                    if constexpr (PAT == AccessPattern::Contiguous) {
                        j1 = i1;
                    } else {
                        j1 = idxDev[i1];
                    }

                    // simple BKMA-like update (read-modify-write)
                    float v = data(i0, j1, i2);
                    scratch(i0, j1, i2) = v + 0.0001f * static_cast<float>(j1);
                });
            }).wait();
        } catch (const sycl::exception &e) {
            state.SkipWithError(e.what());
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
static void BM_LocalMem(benchmark::State &state) {
    const auto data_range = get_range_with_constraint(state.range(0));
    const auto& n0 = data_range.get(0);
    const auto& n1 = data_range.get(1);
    const auto& n2 = data_range.get(2);

    /* SYCL setup */
    auto Q = createSyclQueue(true, state);
    span3d_t data(sycl_alloc(n0*n1*n2, Q), n0, n1, n2);

    // init
    Q.parallel_for(data_range, [=](auto itm){
        const auto& i0 = itm[0]; const auto& i1 = itm[1]; const auto& i2 = itm[2];
        data(i0,i1,i2) = sycl::cos(static_cast<float>(i0 + i1 + i2));
    }).wait();

    // index map for i1 dimension
    const int stride = choose_coprime_stride(n1);
    auto idxHost = make_index_map(PAT, n1, stride);
    int* idxDev = (int*) sycl::malloc_device(sizeof(int)*n1, Q);
    Q.memcpy(idxDev, idxHost.data(), sizeof(int)*n1).wait();

    const size_t w0 = 1, w1 = n1, w2 = 1;
    sycl::range<3> local_range(w0,w1,w2);
    sycl::nd_range ndr(data_range, local_range);

    /* Benchmark */
    for (auto _ : state) {
        try {
            Q.submit([&](sycl::handler &cgh){
                local_acc_1d local_acc(w1, cgh); // scratch along i1 in local memory
                cgh.parallel_for(ndr, [=](auto itm){
                    span1d_t scratch(local_acc.GET_POINTER(), w1);

                    const int i0 = itm.get_global_id(0);
                    const int i1 = itm.get_global_id(1);
                    const int i2 = itm.get_global_id(2);

                    // Stage line [i0,:,i2] into local
                    scratch(i1) = data(i0, i1, i2);
                    itm.barrier(sycl::access::fence_space::local_space);

                    // mapped update in local
                    int j1;
                    if constexpr (PAT == AccessPattern::Contiguous) {
                        j1 = i1;
                    } else {
                        j1 = idxDev[i1];
                    }
                    scratch(j1) = scratch(j1) + 0.0001f * static_cast<float>(j1);
                    itm.barrier(sycl::access::fence_space::local_space);

                    // one thread writes back a value to prevent DCE (cheap)
                    if (i0 == 0 && i1 == 0 && i2 == n2-1)
                        ((span3d_t&)data)(0,0,0) = scratch(0);
                });
            }).wait();
        } catch (const sycl::exception &e) {
            state.SkipWithError(e.what());
        } catch (const std::exception &e) {
            state.SkipWithError(e.what());
            break;
        }
    }

    const auto n_iter = state.iterations();
    // we read+write one value per element (like global path) for fair byte accounting
    state.SetItemsProcessed(n_iter * n0 * n1 * n2);
    state.SetBytesProcessed(n_iter * n0 * n1 * n2 * sizeof(real_t) * 2);
    state.counters.insert({{"gpu", true},{"n0", n0},{"n1", n1},{"n2", n2},{"w0", w0},{"w1", w1},{"w2", w2}});

    sycl::free(idxDev, Q);
    sycl::free(data.data_handle(), Q);
    Q.wait();
}

// Global
static void BM_GlobalMem_Contig(benchmark::State& s){ BM_GlobalMem<AccessPattern::Contiguous>(s); }
static void BM_GlobalMem_Stride(benchmark::State& s){ BM_GlobalMem<AccessPattern::Strided>(s); }
static void BM_GlobalMem_Indirect(benchmark::State& s){ BM_GlobalMem<AccessPattern::Indirect>(s); }

// Local
static void BM_LocalMem_Contig(benchmark::State& s){ BM_LocalMem<AccessPattern::Contiguous>(s); }
static void BM_LocalMem_Stride(benchmark::State& s){ BM_LocalMem<AccessPattern::Strided>(s); }
static void BM_LocalMem_Indirect(benchmark::State& s){ BM_LocalMem<AccessPattern::Indirect>(s); }

// BENCHMARK(BM_GlobalMem_Contig)
//     ->Name("GlobalMem_Contiguous")
//     ->RangeMultiplier(2)->Range(1, 1024)
//     ->UseRealTime()->Unit(benchmark::kMillisecond);

// BENCHMARK(BM_GlobalMem_Stride)
//     ->Name("GlobalMem_Stride")
//     ->RangeMultiplier(2)->Range(1, 1024)
//     ->UseRealTime()->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GlobalMem_Indirect)
    ->Name("GlobalMem_Indirect")
    ->RangeMultiplier(2)->Range(1, 1024)
    ->UseRealTime()->Unit(benchmark::kMillisecond);

// BENCHMARK(BM_LocalMem_Contig)
//     ->Name("LocalMem_Contiguous")
//     ->RangeMultiplier(2)->Range(1, 1024)
//     ->UseRealTime()->Unit(benchmark::kMillisecond);

// BENCHMARK(BM_LocalMem_Stride)
//     ->Name("LocalMem_Stride")
//     ->RangeMultiplier(2)->Range(1, 1024)
//     ->UseRealTime()->Unit(benchmark::kMillisecond);

BENCHMARK(BM_LocalMem_Indirect)
    ->Name("LocalMem_Indirect")
    ->RangeMultiplier(2)->Range(1, 1024)
    ->UseRealTime()->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
