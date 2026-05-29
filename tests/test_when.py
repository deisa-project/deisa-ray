import os
import pytest
import ray

from deisa.ray.types import DeisaArray
import numpy as np
from tests.utils import WorkerSpec

NB_ITERATIONS = 5


@ray.remote(num_cpus=0, max_retries=0, max_calls=1)
def strange_worker(
    *,
    rank: int,
    position: tuple[int, ...],
    chunks_per_dim: tuple[int, ...],
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
    from deisa.ray.comm import init_gloo_comm

    array_names: list[str] = ["array1", "array2"]

    arrays_md = {
        name: {
            "global_shape": tuple(n * c for n, c in zip(chunks_per_dim, chunk_size)),
            "chunk_shape": chunk_size,
            "chunk_position": position,
        }
        for name in array_names
    }

    comm = init_gloo_comm(
        nb_nodes,
        rank,
        "127.0.0.1",
        port,
    )
    client = Bridge(arrays_metadata=arrays_md, comm=comm, _node_id=node_id)

    array = (rank + 1) * np.ones(chunk_size, dtype=dtype)

    for i in range(nb_iterations):
        if i % 2 == 0:
            chunk = i * array
            client.send(array_name=array_names[0], chunk=chunk, timestep=i)
        if i % 2 == 1:
            chunk = i * array
            client.send(array_name=array_names[1], chunk=chunk, timestep=i)
    client.close(timestep=nb_iterations)


@ray.remote(max_retries=0)
def head_script(enable_distributed_scheduling, nb_nodes):
    """The head node checks that the values are correct"""
    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import Window

    os.environ["DEISA_DISTRIBUTED_SCHEDULING"] = "1" if enable_distributed_scheduling else "0"

    d = Deisa()
    or_count = 0
    and_count = 0

    @d.register(Window("array1"), Window("array2"), when="OR")
    def cb_or(array1: list[DeisaArray], array2: list[DeisaArray]):
        nonlocal or_count
        or_count += 1

    @d.register(Window("array1"), Window("array2"), when="AND")
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
def test_and_or_analytics_works_correctly(nb_nodes: int, enable_distributed_scheduling: bool, ray_workflow) -> None:
    ray_workflow.start_head(head_script, enable_distributed_scheduling, nb_nodes)
    ray_workflow.start_simple_workers(
        (
            WorkerSpec(
                rank=rank,
                position=(rank // 2, rank % 2),
                chunks_per_dim=(2, 2),
                chunk_size=(1, 1),
                nb_iterations=NB_ITERATIONS,
                node_id=f"node_{rank % nb_nodes}",
                nb_nodes=4,
            )
            for rank in range(4)
        ),
        worker=strange_worker,
    )

    results = ray_workflow.wait()
    or_count, and_count = results[0]
    assert or_count == 4
    assert and_count == 0

    ray_workflow.assert_scheduling_actor_count(nb_nodes)
