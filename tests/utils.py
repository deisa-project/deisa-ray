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
    nb_nodes: int,
    node_id: str | None = None,
    array_name: str | list[str] = "array",
    dtype: np.dtype = np.int32,  # type: ignore
    **kwargs,
) -> None:
    """Worker node sending chunks of data"""
    from deisa.ray.bridge import Bridge

    if isinstance(array_name, str):
        array_name = [array_name]

    start_iteration = kwargs.get("start_iteration", 0)

    sys_md = {"world_size": nb_nodes, "master_address": "127.0.0.1", "master_port": 29500}
    arrays_md = {
        name: {
            "chunk_shape": chunk_size,
            "nb_chunks_per_dim": chunks_per_dim,
            "nb_chunks_of_node": nb_chunks_of_node,
            "dtype": dtype,
            "chunk_position": position,
        }
        for name in array_name
    }

    client = Bridge(bridge_id=rank, arrays_metadata=arrays_md, system_metadata=sys_md, _node_id=node_id)

    array = (rank + 1) * np.ones(chunk_size, dtype=dtype)

    for i in range(start_iteration, nb_iterations):
        for array_described in list(arrays_md.keys()):
            chunk = i * array
            client.send(array_name=array_described, chunk=chunk, timestep=i, chunked=True)

    client.close(timestep=nb_iterations)


@ray.remote(num_cpus=0, max_retries=0)
def simple_worker_error_test(
    *,
    rank: int,
    position: tuple[int, ...],
    chunks_per_dim: tuple[int, ...],
    nb_chunks_of_node: int,
    chunk_size: tuple[int, ...],
    nb_iterations: int,
    nb_nodes: int,
    node_id: str | None = None,
    array_name: str = "array",
    dtype: np.dtype = np.int32,  # type: ignore
) -> None:
    """Worker node sending chunks of data"""
    from deisa.ray.bridge import Bridge

    sys_md = {"world_size": nb_nodes, "master_address": "127.0.0.1", "master_port": 29500}
    arrays_md = {
        array_name: {
            "chunk_shape": chunk_size,
            "nb_chunks_per_dim": chunks_per_dim,
            "nb_chunks_of_node": nb_chunks_of_node,
            "dtype": dtype,
            "chunk_position": position,
        }
    }

    client = Bridge(bridge_id=rank, arrays_metadata=arrays_md, system_metadata=sys_md, _node_id=node_id)

    array = (rank + 1) * np.ones(chunk_size, dtype=dtype)

    for i in range(nb_iterations):
        chunk = i * array
        if i == nb_iterations / 2:
            client.send(array_name="error", chunk=chunk, timestep=i, chunked=True)
        else:
            client.send(array_name=array_name, chunk=chunk, timestep=i, chunked=True)
    client.close(timestep=nb_iterations)
