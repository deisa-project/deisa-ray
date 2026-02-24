import dask.array as da
import ray
from deisa.ray.types import DeisaArray
from tests.utils import ray_cluster, simple_worker, wait_for_head_node, pick_free_port  # noqa: F401
import pytest

import numpy as np
import shutil
import pathlib
import os

NB_ITERATIONS = 10


@pytest.mark.parametrize(
    "fname, enable_distributed_scheduling",
    [
        ("interesting-event.h5", False),
        ("interesting-event.h5", True),
        ("~/interesting-event.h5", True),
        ("~/interesting-event.h5", False),
    ],
)
def test_dask_save_hdf5(fname, enable_distributed_scheduling, ray_cluster) -> None:  # noqa: F811
    @ray.remote(max_retries=0)
    def head_script(fname, enable_distributed_scheduling) -> None:
        """The head node checks that the values are correct"""
        from deisa.ray.window_handler import Deisa
        from deisa.ray.types import WindowSpec

        import deisa.ray as deisa

        deisa.config.enable_experimental_distributed_scheduling(enable_distributed_scheduling)

        d = Deisa()

        def simulation_callback(array: list[DeisaArray]):
            if array[0].t == 5:
                array[0].to_hdf5(fname)

        d.register_callback(
            simulation_callback,
            [WindowSpec("array")],
        )
        d.execute_callbacks()

    import h5py

    # Check in the correct place.
    full_name = pathlib.Path(fname).expanduser().resolve()

    if os.path.exists(full_name):
        save_dir = full_name.parent
        os.system(f"rm -f {save_dir}/*.h5 {save_dir}/.*h5")

    # Save using relative path
    head_ref = head_script.remote(fname, enable_distributed_scheduling)
    wait_for_head_node()
    port = pick_free_port()

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
                nb_nodes=4,
                port=port,
            )
        )

    ray.get([head_ref] + worker_refs)

    x = h5py.File(full_name)["data"]
    data = da.from_array(x, chunks=2)

    data_sum = data.sum().compute()
    assert 49 < data_sum < 51

    arr = 5 * np.array([[1, 2], [3, 4]])
    assert (data.compute() == arr).all()

    if os.path.exists(full_name):
        save_dir = full_name.parent
        os.system(f"rm -f {save_dir}/*.h5 {save_dir}/.*h5")


@pytest.mark.parametrize(
    "fname, enable_distributed_scheduling",
    [
        ("interesting-event.zarr", False),
        ("~/interesting-event.zarr", False),
        ("interesting-event.zarr", True),
        ("~/interesting-event.zarr", True),
    ],
)
def test_dask_save_zarr(fname, enable_distributed_scheduling, ray_cluster) -> None:  # noqa: F811
    @ray.remote(max_retries=0)
    def head_script(fname, enable_distributed_scheduling) -> None:
        """The head node checks that the values are correct"""
        from deisa.ray.window_handler import Deisa
        from deisa.ray.types import WindowSpec

        import deisa.ray as deisa

        deisa.config.enable_experimental_distributed_scheduling(enable_distributed_scheduling)

        d = Deisa()

        def simulation_callback(array: list[DeisaArray]):
            # If something that we are looking foward happens:
            if array[0].t == 5:
                array[0].to_zarr(fname, component="data")

        d.register_callback(
            simulation_callback,
            [WindowSpec("array")],
        )
        d.execute_callbacks()

    # Check in the correct place.
    full_path = pathlib.Path(fname).expanduser().resolve()

    if os.path.exists(full_path):
        shutil.rmtree(full_path)

    head_ref = head_script.remote(fname, enable_distributed_scheduling)
    wait_for_head_node()
    port = pick_free_port()

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
                nb_nodes=4,
                port=port,
            )
        )

    ray.get([head_ref] + worker_refs)

    data = da.from_zarr(full_path, component="data")

    data_sum = data.sum().compute()
    assert 49 < data_sum < 51

    arr = 5 * np.array([[1, 2], [3, 4]])
    assert (data.compute() == arr).all()

    if os.path.exists(full_path):
        shutil.rmtree(full_path)


# TODO use also NetCDF
