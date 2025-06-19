#!/bin/bash

export REP=50
export HW=pvc
export IMPL=acpp
export EXE="/home/ac.amillan/source/phd-experiments/build_${IMPL}_${HW}/main"
export OUTFILE="/home/ac.amillan/source/phd-experiments/out/memory-spaces/${IMPL}_${REP}_reps_${HW}.json"

$EXE --benchmark_counters_tabular=true \
     --benchmark_report_aggregates_only=true \
     --benchmark_min_warmup_time=1 \
     --benchmark_format=json \
     --benchmark_repetitions=$REP \
     --benchmark_out=$OUTFILE
