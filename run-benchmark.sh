#!/bin/bash

export REP=50
export EXE="/home/ac.amillan/source/phd-experiments/build_pvc/main"
export OUTFILE="/home/ac.amillan/source/phd-experiments/out/memory-spaces/dpcpp_${REP}_reps_pvc.json"

$EXE --benchmark_counters_tabular=true \
     --benchmark_report_aggregates_only=true \
     --benchmark_min_warmup_time=1 \
     --benchmark_format=json \
     --benchmark_repetitions=$REP \
     --benchmark_out=$OUTFILE
