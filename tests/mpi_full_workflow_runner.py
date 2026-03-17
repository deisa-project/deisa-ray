from __future__ import annotations

import logging

import numpy as np
import ray

from deisa.ray.bridge import Bridge
from deisa.ray.comm import MPICommAdapter

NB_ITERATIONS = 5


@ray.remote(max_retries=0)
def head_script() -> None:
    from deisa.ray.types import DeisaArray, WindowSpec
    from deisa.ray.window_handler import Deisa

    d = Deisa()

    @d.callback(WindowSpec("array"))
    def simulation_callback(array: list[DeisaArray]):
        x = array[0].dask.sum().compute()
        assert x == 10 * array[0].t

    d.execute_callbacks()


def main() -> None:
    from mpi4py import MPI

    mpi_comm = MPICommAdapter(MPI.COMM_WORLD)

    if not ray.is_initialized():
        ray.init(address="auto", log_to_driver=False, logging_level=logging.ERROR)

    head_ref = head_script.remote() if mpi_comm.rank == 0 else None

    arrays_md = {
        "array": {
            "chunk_shape": (1, 1),
            "nb_chunks_per_dim": (2, 2),
            "nb_chunks_of_node": mpi_comm.world_size,
            "dtype": np.int32,
            "chunk_position": (mpi_comm.rank // 2, mpi_comm.rank % 2),
        }
    }
    bridge = Bridge(
        bridge_id=mpi_comm.rank,
        arrays_metadata=arrays_md,
        system_metadata=None,
        comm=mpi_comm,
        _node_id="mpi-node",
    )

    array = (mpi_comm.rank + 1) * np.ones((1, 1), dtype=np.int32)

    for timestep in range(NB_ITERATIONS):
        bridge.send(array_name="array", chunk=timestep * array, timestep=timestep)

    assert bridge.close(timestep=NB_ITERATIONS) == NB_ITERATIONS

    mpi_comm.barrier()

    if mpi_comm.rank == 0 and head_ref is not None:
        ray.get(head_ref)

    ray.shutdown()


if __name__ == "__main__":
    main()
