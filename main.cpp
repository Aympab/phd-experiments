#include "utils.hpp"

static void
BM_GlobalMem_Stride(benchmark::State &state) {
    const auto data_range = get_range_with_constraint(state.range(0));

    const auto& n0 = data_range.get(0);
    const auto& n1 = data_range.get(1);
    const auto& n2 = data_range.get(2);

    /* SYCL setup */
    auto Q = createSyclQueue(true, state);
    span3d_t data(sycl_alloc(n0*n1*n2, Q), n0, n1, n2);
    span3d_t scratch(sycl_alloc(n0*n1*n2, Q), n0, n1, n2);

    Q.wait();
    Q.parallel_for(data_range, [=](auto itm){
        const auto& i0 = itm[0];
        const auto& i1 = itm[1];
        const auto& i2 = itm[2];

        data(i0, i1, i2) = sycl::cos(static_cast<float>(i0 + i1 + i2));
        scratch(i0, i1, i2) = 0;
    }).wait();

    const auto w0 = 1;
    const auto w1 = n1;
    const auto w2 = 1;
    sycl::range<3> local_range(w0,w1,w2);
    sycl::nd_range ndr(data_range, local_range);

    /* Benchmark */
    for (auto _ : state) {
        try {
            Q.submit([&](sycl::handler &cgh){
                cgh.parallel_for(ndr, [=](auto itm){
                    const auto& i0 = itm.get_global_id(0);
                    const auto& i1 = itm.get_global_id(1);
                    const auto& i2 = itm.get_global_id(2);

                    scratch(i0, i1, i2) = data(i0, i1, i2);
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
    state.SetBytesProcessed(n_iter * n0 * n1 * n2 * sizeof(real_t)*2);

          /* Benchmark infos */
        state.counters.insert({
            {"gpu", true},
            {"n0", n0},
            {"n1", n1},
            {"n2", n2},
            {"w0", w0},
            {"w1", w1},
            {"w2", w2},
        });

    sycl::free(data.data_handle(), Q);
    sycl::free(scratch.data_handle(), Q);
    Q.wait();
}

static void
BM_LocalMem_Stride(benchmark::State &state) {
    const auto data_range = get_range_with_constraint(state.range(0));

    const auto& n0 = data_range.get(0);
    const auto& n1 = data_range.get(1);
    const auto& n2 = data_range.get(2);

    /* SYCL setup */
    auto Q = createSyclQueue(true, state);
    span3d_t data(sycl_alloc(n0*n1*n2, Q), n0, n1, n2);
    Q.wait();
    Q.parallel_for(data_range, [=](auto itm){
        const auto& i0 = itm[0];
        const auto& i1 = itm[1];
        const auto& i2 = itm[2];

        data(i0, i1, i2) = sycl::cos(static_cast<float>(i0 + i1 + i2));
    }).wait();

    const auto w0 = 1;
    const auto w1 = n1;
    const auto w2 = 1;
    sycl::range<3> local_range(w0,w1,w2);
    sycl::nd_range ndr(data_range, local_range);

    /* Benchmark */
    for (auto _ : state) {
        try {
            Q.submit([&](sycl::handler &cgh){
            local_acc_1d local_acc(w1, cgh);
                cgh.parallel_for(ndr, [=](auto itm){
                    span1d_t scratch(local_acc.GET_POINTER(), w1);

                    const auto& i0 = itm.get_global_id(0);
                    const auto& i1 = itm.get_global_id(1);
                    const auto& i2 = itm.get_global_id(2);

                    scratch(i1) = data(i0, i1, i2);

                    if(i0 == 0 && i1 == 0 && i2 == n2-1)
                        data(0,0,0) = scratch(i1);
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
    state.SetBytesProcessed(n_iter * n0 * n1 * n2 * sizeof(real_t)*2);

          /* Benchmark infos */
        state.counters.insert({
            {"gpu", true},
            {"n0", n0},
            {"n1", n1},
            {"n2", n2},
            {"w0", w0},
            {"w1", w1},
            {"w2", w2},
        });

    sycl::free(data.data_handle(), Q);
    Q.wait();
}

// ==========================================
BENCHMARK(BM_LocalMem_Stride)
    ->Name("LocalMem_Stride")
    ->RangeMultiplier(2)->Range(1, 1024)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GlobalMem_Stride)
    ->Name("GlobalMem_Stride")
    ->RangeMultiplier(2)->Range(1, 1024)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

// ==========================================
// ==========================================
BENCHMARK_MAIN();
