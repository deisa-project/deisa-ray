# Getting started example

A minimal end-to-end DEISA example that runs a small MPI simulation, streams a
field to Ray, and runs an in situ analytics callback that plots it.

## What it does

- `simulation.py` decomposes a `64 x 64` temperature field over a `2 x 2` MPI
  grid (4 ranks). Each rank owns one `32 x 32` chunk and sends its local chunk
  for 5 timesteps through a `Bridge`.
- `analysis.py` registers a `Deisa` callback for the `temperature` array. For
  every timestep it prints the sum of the field and writes a `heat-<t>.png`
  image to the current directory.
- `launch.sh` starts a single-node Ray cluster (one head + one worker) on
  `127.0.0.1`, launches the analytics process, then runs the 4-rank simulation.

## Requirements

- `deisa-ray` installed (`pip install deisa-ray`).
- MPI and `mpi4py` available on the machine.
- `numpy` and `matplotlib` (pulled in by `deisa-ray`).
- Exactly 4 MPI ranks for the simulation, matching `mpi_parallelism = (2, 2)`
  in `simulation.py`.

> **Note:** do not install or use the `distributed` library with `deisa-ray`.
> DEISA installs its own Dask-on-Ray scheduler and the two are not compatible.

## Run it

The launcher handles the Ray cluster and the process orchestration:

```bash
cd example/getting-started
./launch.sh
```

When it finishes you should see a `latest temperature t=<t>, sum=<...>` line per
timestep and `heat-0.png` ... `heat-4.png` in the directory.

## Run it manually

If you already have a Ray cluster, or want to run the steps yourself:

```bash
# 1. Start a Ray head node (once) and join a worker.
ray start --head --port=6379
ray start --address=127.0.0.1:6379

# 2. Start analytics (one process).
mpirun -np 1 python3 analysis.py &

# 3. Run the simulation (four ranks).
mpirun -np 4 python3 simulation.py

# 4. Shut the cluster down when done.
ray stop
```

The simulation asserts that it is launched with exactly 4 ranks. To use a
different decomposition, update `mpi_parallelism` in `simulation.py` and launch
with `mpi_parallelism[0] * mpi_parallelism[1]` ranks.

## See also

- `docs/source/quick_start.rst` for the concepts behind this example.
- `docs/source/analytics.rst` for more callback patterns and windows.
