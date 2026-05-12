import time

import numpy as np
import pytest
import ray
import socket


def pick_free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# @pytest.fixture(scope = "session")
@pytest.fixture()
def ray_cluster():
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    try:
        yield ray.get_runtime_context().gcs_address
    finally:
        if ray.is_initialized():
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
    chunk_size: tuple[int, ...],
    nb_iterations: int,
    nb_nodes: int,
    port: int,
    node_id: str | None = None,
    array_name: str | list[str] = "array",
    dtype: np.dtype = np.int32,  # type: ignore
    _sleep_b4_send=0,
    _sleep_intra_send=0,
    **kwargs,
) -> None:
    """Worker node sending chunks of data"""
    from deisa.ray.bridge import Bridge
    from deisa.ray.comm import init_gloo_comm

    if isinstance(array_name, str):
        array_name = [array_name]

    start_iteration = kwargs.get("start_iteration", 0)

    sys_md = {"world_size": nb_nodes, "master_address": "127.0.0.1", "master_port": port}
    arrays_md = {
        name: {
            "global_shape": tuple(n * c for n, c in zip(chunks_per_dim, chunk_size)),
            "chunk_shape": chunk_size,
            "chunk_position": position,
        }
        for name in array_name
    }

    comm = init_gloo_comm(
        sys_md["world_size"],
        rank,
        sys_md["master_address"],
        sys_md["master_port"],
    )
    client = Bridge(arrays_metadata=arrays_md, comm=comm, system_metadata=sys_md, _node_id=node_id)

    array = (rank + 1) * np.ones(chunk_size, dtype=dtype)

    time.sleep(_sleep_b4_send)
    for i in range(start_iteration, nb_iterations):
        time.sleep(_sleep_intra_send)
        for array_described in list(arrays_md.keys()):
            chunk = i * array
            client.send(array_name=array_described, chunk=chunk, timestep=i)

    client.close(timestep=nb_iterations)


@ray.remote(num_cpus=0, max_retries=0)
def simple_worker_error_test(
    *,
    rank: int,
    position: tuple[int, ...],
    chunks_per_dim: tuple[int, ...],
    chunk_size: tuple[int, ...],
    nb_iterations: int,
    nb_nodes: int,
    port: int,
    node_id: str | None = None,
    array_name: str = "array",
    dtype: np.dtype = np.int32,  # type: ignore
) -> None:
    """Worker node sending chunks of data"""
    from deisa.ray.bridge import Bridge
    from deisa.ray.comm import init_gloo_comm

    sys_md = {"world_size": nb_nodes, "master_address": "127.0.0.1", "master_port": port}
    arrays_md = {
        array_name: {
            "global_shape": tuple(n * c for n, c in zip(chunks_per_dim, chunk_size)),
            "chunk_shape": chunk_size,
            "chunk_position": position,
        }
    }

    comm = init_gloo_comm(
        sys_md["world_size"],
        rank,
        sys_md["master_address"],
        sys_md["master_port"],
    )
    client = Bridge(arrays_metadata=arrays_md, comm=comm, system_metadata=sys_md, _node_id=node_id)

    array = (rank + 1) * np.ones(chunk_size, dtype=dtype)

    for i in range(nb_iterations):
        chunk = i * array
        if i == nb_iterations // 2:
            client.send(array_name="error", chunk=chunk, timestep=i)
        else:
            client.send(array_name=array_name, chunk=chunk, timestep=i)
    client.close(timestep=nb_iterations)
