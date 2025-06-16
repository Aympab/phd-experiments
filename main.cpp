#include "utils.hpp"

static void
BM_GlobalMem_Stride(benchmark::State &state) {}

static void
BM_LocalMem_Stride(benchmark::State &state) {
    const auto data_range = get_range_with_constraint(state.range(0));

    const auto& n0 = data_range.get(0);
    const auto& n1 = data_range.get(1);
    const auto& n2 = data_range.get(2);

    const auto w0 = 1;
    const auto w1 = 128;
    const auto w2 = 1;

    /* SYCL setup */
    auto Q = createSyclQueue(true, state);
    span3d_t data(sycl_alloc(n0*n1*n2, Q), n0, n1, n2);
    Q.wait();

    /* Benchmark */
    for (auto _ : state) {
        try {
            // bkma_run_function(Q, data, solver, optim_params, span3d_t{});
            Q.wait();
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
BENCHMARK(BM_Advection)
    ->Name("main-BKM-bench")
    ->RangeMultiplier(2)->Range(1, 1024)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

// ==========================================
// ==========================================
BENCHMARK_MAIN();
