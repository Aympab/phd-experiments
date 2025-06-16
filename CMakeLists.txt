cmake_minimum_required (VERSION 3.20)

project(phd-experiments LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(benchmark REQUIRED)

if(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM") OR (NOT ${DPCPP_FSYCL_TARGETS} STREQUAL "") OR (NOT ${BUILD_WITH_INTEL_LLVM} STREQUAL ""))
    set(SYCL_IS_ACPP OFF)
else()
    set(SYCL_IS_ACPP ON)
    find_package(AdaptiveCpp REQUIRED)
endif()

add_library(dpcpp_opt INTERFACE)
if(NOT SYCL_IS_ACPP)
    message(STATUS "Setting flags for IntelLLVM compiler")
    target_link_options(dpcpp_opt INTERFACE -fno-sycl-id-queries-fit-in-int -fsycl ${DPCPP_FSYCL_TARGETS})
    target_compile_options(dpcpp_opt INTERFACE -fno-sycl-id-queries-fit-in-int -fsycl ${DPCPP_FSYCL_TARGETS})
else()
    message(STATUS "To add correct flags for DPCPP compiler, use the "
    "DPCPP_FSYCL_TARGETS' variable. Example: "
    "-DDPCPP_FSYCL_TARGETS='-fsycl-targets=nvptx64-nvidia-cuda'")
endif()

function(add_bench_executable name)
    add_executable(${name} ${name}.cpp)
  
    target_link_libraries(${name}
    PUBLIC
    benchmark::benchmark)
  
    target_include_directories(${name}
      PUBLIC
      ${CMAKE_SOURCE_DIR}/src/tools
      ${CMAKE_SOURCE_DIR}/src/core
      ${CMAKE_SOURCE_DIR}/src/solvers
      ${CMAKE_SOURCE_DIR}/src
    )
  
    if(SYCL_IS_ACPP)
        add_sycl_to_target(TARGET ${name})
    else()
        target_link_libraries(${name} PUBLIC dpcpp_opt)
    endif()
endfunction()

add_subdirectory(tpls/mdspan)
add_bench_executable(main)
