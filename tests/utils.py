import time

import numpy as np
import pytest
import ray


# @pytest.fixture(scope = "session")
@pytest.fixture()
def ray_cluster():
    """Start a Ray cluster for this test"""
    ray.init()
    yield
    ray.shutdown()


def wait_for_head_node() -> None:
    """Wait until the head node is ready"""
    while True:
        try:
            a = ray.get_actor("simulation_head", namespace="deisa_ray")
            ray.get(a.ready.remote())
            return
        except ValueError:
            time.sleep(0.1)


@ray.remote(num_cpus=0, max_retries=0)
def simple_worker(
    *,
    rank: int,
    position: tuple[int, ...],
    chunks_per_dim: tuple[int, ...],
    nb_chunks_of_node: int,
    chunk_size: tuple[int, ...],
    nb_iterations: int,
    node_id: str | None = None,
    array_name: str = "array",
    dtype: np.dtype = np.int32,  # type: ignore
) -> None:
    """Worker node sending chunks of data"""
    from deisa.ray.bridge import Bridge

    client = Bridge(_node_id=node_id)

    array = (rank + 1) * np.ones(chunk_size, dtype=dtype)

    for i in range(nb_iterations):
        client.add_chunk(array_name, position, chunks_per_dim, nb_chunks_of_node, i, i * array, store_externally=False)
