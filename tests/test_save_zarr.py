import dask.array as da
import ray
from deisa.ray.types import DeisaArray
from tests.utils import ray_cluster, simple_worker, wait_for_head_node  # noqa: F401
import pytest

import numpy as np
import zarr
import shutil
import os

NB_ITERATIONS = 10
ZARR_PATH = "/tmp/test_deisa_simulation.zarr"

@ray.remote(max_retries=0)
def head_script(enable_distributed_scheduling) -> None:
    """The head node checks that the values are correct"""
    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import WindowSpec

    import deisa.ray as deisa

    deisa.config.enable_experimental_distributed_scheduling(enable_distributed_scheduling)

    d = Deisa(n_sim_nodes=4)

    def simulation_callback(array: list[DeisaArray]):
        dask_arr = array[0].dask.persist()
        timestep = array[0].t

        da.to_zarr(
            dask_arr,
            ZARR_PATH,
            component=str(timestep),
            compute=True
        )


    d.register_callback(
        simulation_callback,
        [WindowSpec("array")],
    )
    d.execute_callbacks()


@pytest.mark.parametrize("enable_distributed_scheduling", [True, False])
def test_dask_save_zarr(enable_distributed_scheduling, ray_cluster) -> None:  # noqa: F811

    if os.path.exists(ZARR_PATH):
        shutil.rmtree(ZARR_PATH)

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

    # Check if saved correctly
    z_group = zarr.open_group(ZARR_PATH, mode='r')
    assert len(z_group) == NB_ITERATIONS

    for timestep in range(NB_ITERATIONS):

        data = da.from_zarr(ZARR_PATH, component=str(timestep)).compute()

        assert data.sum() == timestep * 10

        arr = timestep * np.array([[1, 2], [3, 4]])
        assert (arr == np.array(data)).all()

    if os.path.exists(ZARR_PATH):
         shutil.rmtree(ZARR_PATH)
