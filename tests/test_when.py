import pytest
import ray

from deisa.ray.types import DeisaArray
from tests.utils import wait_for_head_node  # noqa: F401
import numpy as np
from tests.utils import pick_free_port

NB_ITERATIONS = 5


@ray.remote(num_cpus=0, max_retries=0)
def strange_worker(
    *,
    rank: int,
    position: tuple[int, ...],
    chunks_per_dim: tuple[int, ...],
    nb_chunks_of_node: int,
    chunk_size: tuple[int, ...],
    nb_iterations: int,
    nb_nodes: int,
    port: int,
    node_id: str | None = None,
    dtype: np.dtype = np.int32,  # type: ignore
    **kwargs,
) -> None:
    """Strange worker that sends nodes out of order!"""
    from deisa.ray.bridge import Bridge

    array_names: list[str] = ["array1", "array2"]

    sys_md = {"world_size": nb_nodes, "master_address": "127.0.0.1", "master_port": port}
    arrays_md = {
        name: {
            "chunk_shape": chunk_size,
            "nb_chunks_per_dim": chunks_per_dim,
            "nb_chunks_of_node": nb_chunks_of_node,
            "dtype": dtype,
            "chunk_position": position,
        }
        for name in array_names
    }

    client = Bridge(bridge_id=rank, arrays_metadata=arrays_md, system_metadata=sys_md, _node_id=node_id)

    array = (rank + 1) * np.ones(chunk_size, dtype=dtype)

    for i in range(nb_iterations):
        if i % 2 == 0:
            chunk = i * array
            client.send(array_name=array_names[0], chunk=chunk, timestep=i, chunked=True)
        if i % 2 == 1:
            chunk = i * array
            client.send(array_name=array_names[1], chunk=chunk, timestep=i, chunked=True)
    client.close(timestep=nb_iterations)


@ray.remote(max_retries=0)
def head_script(enable_distributed_scheduling, nb_nodes):
    """The head node checks that the values are correct"""
    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import WindowSpec
    import deisa.ray as deisa

    deisa.config.enable_experimental_distributed_scheduling(enable_distributed_scheduling)

    d = Deisa()
    or_count = 0
    and_count = 0

    @d.callback(WindowSpec("array1"), WindowSpec("array2"), when="OR")
    def cb_or(array1: list[DeisaArray], array2: list[DeisaArray]):
        nonlocal or_count
        or_count += 1

    @d.callback(WindowSpec("array1"), WindowSpec("array2"), when="AND")
    def cb_and(array1: list[DeisaArray], array2: list[DeisaArray]):
        nonlocal and_count
        and_count += 1

    d.execute_callbacks()
    return or_count, and_count


# Exposure pattern (NB_ITERATIONS = 5):
#
#   t:        0   1   2   3   4
#   array1:   X       X       X
#   array2:       X       X
#
# Arrays are never exposed on the same timestep → AND = 0.
#
# OR requires "any new" but only after both arrays have been seen.
# At t=0 only array1 is seen → suppressed.
# From t=1..4 exactly one array updates each step → 4 calls.
@pytest.mark.parametrize(
    "nb_nodes, enable_distributed_scheduling",
    [
        (4, True),
        (4, False),
    ],
)
def test_and_or_analytics_works_correctly(nb_nodes: int, enable_distributed_scheduling: bool, ray_cluster) -> None:  # noqa: F811
    port = pick_free_port()
    head_ref = head_script.remote(enable_distributed_scheduling, nb_nodes)
    wait_for_head_node()

    worker_refs = []
    for rank in range(4):
        worker_refs.append(
            strange_worker.remote(
                rank=rank,
                position=(rank // 2, rank % 2),
                chunks_per_dim=(2, 2),
                nb_chunks_of_node=4 // nb_nodes,
                chunk_size=(1, 1),
                nb_iterations=NB_ITERATIONS,
                node_id=f"node_{rank % nb_nodes}",
                nb_nodes=4,
                port=port,
            )
        )

    results = ray.get([head_ref] + worker_refs)
    or_count, and_count = results[0]
    assert or_count == 4
    assert and_count == 0

    # Check that the right number of scheduling actors were created
    simulation_head = ray.get_actor("simulation_head", namespace="deisa_ray")
    assert len(ray.get(simulation_head.list_scheduling_actors.remote())) == nb_nodes
