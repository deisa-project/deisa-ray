import dask.array as da
import ray
from deisa.ray.types import DeisaArray
from tests.utils import ray_cluster, simple_worker, wait_for_head_node  # noqa: F401
import pytest

import numpy as np
import shutil
import os

NB_ITERATIONS = 10
HDF5_PATH = "interesting-event.h5"
ZARR_PATH = "interesting-event.zarr"


@pytest.mark.parametrize("enable_distributed_scheduling", [True, False])
def test_dask_save_hdf5(enable_distributed_scheduling, ray_cluster) -> None:  # noqa: F811
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

            # If something interesting happens:
            if 49 < arr_sum < 51:
                array[0].to_hdf5(HDF5_PATH)

        d.register_callback(
            simulation_callback,
            [WindowSpec("array")],
        )
        d.execute_callbacks()

    import h5py

    if os.path.exists(HDF5_PATH):
        os.system("rm -f *.h5 .*.h5")

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
    data = da.from_array(x, chunks=2)

    data_sum = data.sum().compute()
    assert 49 < data_sum < 51

    arr = 5 * np.array([[1, 2], [3, 4]])
    assert (data.compute() == arr).all()

    if os.path.exists(HDF5_PATH):
        os.system("rm -f *.h5 .*.h5")


@pytest.mark.parametrize("enable_distributed_scheduling", [True, False])
def test_dask_save_zarr(enable_distributed_scheduling, ray_cluster) -> None:  # noqa: F811
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
                array[0].to_zarr(ZARR_PATH, component="data")

        d.register_callback(
            simulation_callback,
            [WindowSpec("array")],
        )
        d.execute_callbacks()

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

    data = da.from_zarr(ZARR_PATH, component="data")

    data_sum = data.sum().compute()
    assert 49 < data_sum < 51

    arr = 5 * np.array([[1, 2], [3, 4]])
    assert (data.compute() == arr).all()

    if os.path.exists(ZARR_PATH):
        shutil.rmtree(ZARR_PATH)


# TODO use also NetCDF
