import dask.array as da
import ray
from deisa.ray.types import DeisaArray
from tests.utils import ray_cluster, simple_worker, wait_for_head_node  # noqa: F401
import pytest

import numpy as np
import h5py
import os

NB_ITERATIONS = 10
HDF5_PATH = "amazing-event.h5"


@ray.remote(max_retries=0)
def head_script(enable_distributed_scheduling) -> None:
    """The head node checks that the values are correct"""
    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import WindowSpec

    import deisa.ray as deisa

    deisa.config.enable_experimental_distributed_scheduling(enable_distributed_scheduling)

    d = Deisa(n_sim_nodes=4)

    def simulation_callback(array: list[DeisaArray]):
        arr_sum = array[0].dask.sum().compute()

        # If something that we are looking foward happens:
        if 49 < arr_sum < 51:
            array[0].to_hdf5("amazing-event.h5")

    d.register_callback(
        simulation_callback,
        [WindowSpec("array")],
    )
    d.execute_callbacks()


@pytest.mark.parametrize("enable_distributed_scheduling", [True, False])
def test_dask_save_hdf5(enable_distributed_scheduling, ray_cluster) -> None:  # noqa: F811
    if os.path.exists(HDF5_PATH):
        os.system("rm -f *amazing-event.h5 .amazing-event*.h5")

    head_ref = head_script.remote(enable_distributed_scheduling)
    wait_for_head_node()

    worker_refs = []
    for rank in range(4):
        worker_refs.append(
            simple_worker.remote(
                rank=rank,
                position=(rank // 2, rank % 2),
                chunks_per_dim=(2, 2),
                nb_chunks_of_node=1,
                chunk_size=(1, 1),
                nb_iterations=NB_ITERATIONS,
                node_id=f"node_{rank}",
            )
        )

    ray.get([head_ref] + worker_refs)

    x = h5py.File(HDF5_PATH)["data"]
    a = da.from_array(x, chunks=2)

    assert (a.compute() == np.array([[5, 10], [15, 20]])).all()
    assert 49 < a.sum().compute() < 51

    if os.path.exists(HDF5_PATH):
        os.system("rm -f *amazing-event.h5 .amazing-event*.h5")
