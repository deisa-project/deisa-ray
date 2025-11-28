import os
import random

import dask.array as da
import pytest
import ray

from tests.utils import ray_cluster, simple_worker, wait_for_head_node  # noqa: F401

NB_ITERATIONS = 100  # Should be enough to saturate the memory in case the chunks are not released


@ray.remote(max_retries=0)
def head_script() -> None:
    """The head node checks that the values are correct"""
    from deisa.ray.head_node import init
    from deisa.ray.window_api import run_simulation
    from deisa.ray.types import WindowArrayDefinition

    init()

    def simulation_callback(array: da.Array, timestep: int):
        pass

    run_simulation(
        simulation_callback,
        [WindowArrayDefinition("array")],
        max_iterations=NB_ITERATIONS,
    )


@pytest.fixture
def ray_spilling_cluster():
    spilling_path = f"/tmp/deisa_ray_spilling_test_{random.randint(0, 2**128 - 1)}"

    ray.init(object_store_memory=100 * 1024 * 1024, object_spilling_directory=spilling_path)
    yield spilling_path
    ray.shutdown()


def test_memory_release(ray_spilling_cluster: str) -> None:  # noqa: F811
    """
    Perform a long simulation with a small object store spilling to disk. If the
    memory is not released correctly, the test will detect spilled objects on disk and
    fail.
    """
    head_ref = head_script.remote()
    wait_for_head_node()

    worker = simple_worker.remote(
        rank=0,
        position=(0, 0),
        chunks_per_dim=(1, 1),
        nb_chunks_of_node=1,
        chunk_size=(1024, 1024),
        nb_iterations=NB_ITERATIONS,
    )

    ray.get([head_ref, worker])

    # Make sure Ray didn't spill anything to disk
    total_size = 0  # in bytes
    for dirpath, dirnames, filenames in os.walk(ray_spilling_cluster):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.isfile(filepath):
                total_size += os.path.getsize(filepath)

    assert total_size == 0
