#!/bin/bash

cmake .. -DCMAKE_CXX_COMPILER=clang++ -Dbenchmark_DIR=/home/ac.amillan/source/parallel-advection/thirdparty/benchmark/build -DDPCPP_FSYCL_TARGETS='-fsycl-targets=nvidia_gpu_sm_90'