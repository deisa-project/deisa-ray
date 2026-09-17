#!/usr/bin/env bash

export DEISA_DASK_SCHEDULER_ADDRESS="tcp://localhost:8786"

mpirun -np 1 ray start --node-ip-address=127.0.0.1 --head --port=6379 &
head_pid=$!

sleep 5

mpirun -np 1 ray start --address=127.0.0.1:6379 &
worker_pid=$!

sleep 2

export DEISA_DISTRIBUTED_SCHEDULING_ENV=1
mpirun -np 1 python3 analysis.py &
analysis_pid=$!

mpirun -np 4 python3 simulation.py
simulation_pid=$!

echo "kill worker PID: $worker_pid"
kill $worker_pid
echo "waiting for worker, analysis and simulation PIDs to finish"
wait $worker_pid $analysis_pid $simulation_pid
echo "kill scheduler PID: $head_pid"
ray stop

echo "launcher is done."
