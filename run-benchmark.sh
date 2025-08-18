#!/bin/bash

export HW=pvc
export IMPL=dpcpp
export REP=10
export EXE="/home/ac.amillan/source/phd-experiments/build_${IMPL}_${HW}/main"
export OUTFILE="/home/ac.amillan/source/phd-experiments/out/memory-spaces-new/strided/${IMPL}_${HW}.json"

$EXE --benchmark_counters_tabular=true \
     --benchmark_report_aggregates_only=true \
     --benchmark_min_warmup_time=1 \
     --benchmark_format=json \
     --benchmark_repetitions=$REP \
     --benchmark_out=$OUTFILE
