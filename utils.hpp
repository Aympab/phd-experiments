#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>
#include <experimental/mdspan>

// =============================================
//                    Types
// =============================================
using real_t = double;

using span0d_t =
    std::experimental::mdspan<real_t, std::experimental::dextents<size_t, 0>,
                              std::experimental::layout_right>;
using span1d_t =
    std::experimental::mdspan<real_t, std::experimental::dextents<size_t, 1>,
                              std::experimental::layout_right>;
using span2d_t =
    std::experimental::mdspan<real_t, std::experimental::dextents<size_t, 2>,
                              std::experimental::layout_right>;
using span3d_t =
    std::experimental::mdspan<real_t, std::experimental::dextents<size_t, 3>,
                              std::experimental::layout_right>;

using local_acc_1d = sycl::local_accessor<real_t, 1>;

using extents_t =
    std::experimental::extents<std::size_t, std::experimental::dynamic_extent,
                               std::experimental::dynamic_extent,
                               std::experimental::dynamic_extent>;


// =============================================
//                   SYCL utils
// =============================================
#ifdef SYCL_IMPLEMENTATION_ONEAPI
#define GET_POINTER get_multi_ptr<sycl::access::decorated::no>().get
#else
#define GET_POINTER get_pointer
#endif

[[nodiscard]] inline auto
sycl_alloc(size_t size, sycl::queue &q) {
    return sycl::malloc_shared<real_t>(size, q);
}

[[nodiscard]] inline sycl::queue
createSyclQueue(const bool run_on_gpu, benchmark::State &state) {
    sycl::device d;

    if (run_on_gpu)
        try {
            d = sycl::device{sycl::gpu_selector_v};
        } catch (const sycl::exception e) {
            state.SkipWithError(
                "GPU was requested but none is available, skipping benchmark.");
        }
    else
        d = sycl::device{sycl::cpu_selector_v};
    return sycl::queue{d};
}   // end createSyclQueue

// =============================================
//               Benchmark utils
// =============================================
[[nodiscard]] inline sycl::range<3>
get_range_with_constraint(const size_t n2) {
    const auto n_total = 4194304*2; //2**22
    const auto n1 = 128*2;
    const auto n0 = n_total/n1/n2;
    return sycl::range<3>(n0, n1, n2);
}
