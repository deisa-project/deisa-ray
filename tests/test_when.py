import os

import numpy as np
import pytest
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from deisa.ray.types import DeisaArray
from tests.utils import pick_free_port, ray_multinode_cluster, wait_for_head_node  # noqa: F401

NB_ITERATIONS = 5


@ray.remote(num_cpus=0, max_retries=0)
def bridge_script(*, rank: int, port: int) -> None:
    from deisa.ray.bridge import Bridge
    from tests.comm_utils import init_gloo_comm

    array_names: list[str] = ["array1", "array2"]

    arrays_md = {
        name: {
            "global_shape": (1, 2),
            "chunk_shape": (1, 1),
            "chunk_position": (0, rank),
        }
        for name in array_names
    }

    comm = init_gloo_comm(
        2,
        rank,
        "127.0.0.1",
        port,
    )
    bridge = Bridge(arrays_metadata=arrays_md, comm=comm, _node_id=f"node_{rank}")

    array = (rank + 1) * np.ones((1, 1), dtype=np.int32)

    for timestep in range(NB_ITERATIONS):
        if timestep % 2 == 0:
            bridge.send(array_name=array_names[0], chunk=timestep * array, timestep=timestep)
        if timestep % 2 == 1:
            bridge.send(array_name=array_names[1], chunk=timestep * array, timestep=timestep)
    bridge.close(timestep=NB_ITERATIONS)


@ray.remote(num_cpus=0, max_retries=0)
def bridge_script_all_arrays(*, rank: int, port: int) -> None:
    from deisa.ray.bridge import Bridge
    from tests.comm_utils import init_gloo_comm

    array_names: list[str] = ["array1", "array2"]

    arrays_md = {
        name: {
            "global_shape": (1, 2),
            "chunk_shape": (1, 1),
            "chunk_position": (0, rank),
        }
        for name in array_names
    }

    comm = init_gloo_comm(
        2,
        rank,
        "127.0.0.1",
        port,
    )
    bridge = Bridge(arrays_metadata=arrays_md, comm=comm, _node_id=f"node_{rank}")

    array = (rank + 1) * np.ones((1, 1), dtype=np.int32)

    for timestep in range(NB_ITERATIONS):
        for array_name in array_names:
            bridge.send(array_name=array_name, chunk=timestep * array, timestep=timestep)
    bridge.close(timestep=NB_ITERATIONS)


@ray.remote(max_retries=0)
def head_script(enable_distributed_scheduling):
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


def get_node_ids(ray_multinode_cluster):
    cluster = ray_multinode_cluster["cluster"]
    head_node_id = None
    worker_node_ids = []
    for node in cluster.list_all_nodes():
        if node.is_head():
            head_node_id = node.node_id
        else:
            worker_node_ids.append(node.node_id)
    return head_node_id, worker_node_ids


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
    "enable_distributed_scheduling",
    [True, False],
)
def test_and_or_analytics_works_correctly(enable_distributed_scheduling: bool, ray_multinode_cluster) -> None:  # noqa: F811
    head_node_id, worker_node_ids = get_node_ids(ray_multinode_cluster)

    head_ref = head_script.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=head_node_id,
            soft=False,
        ),
    ).remote(enable_distributed_scheduling)
    wait_for_head_node()
    port = pick_free_port()

    worker_refs = [
        bridge_script.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=worker_node_ids[rank],
                soft=False,
            ),
        ).remote(rank=rank, port=port)
        for rank in range(2)
    ]

    results = ray.get([head_ref] + worker_refs)
    or_count, and_count = results[0]
    assert or_count == 4
    assert and_count == 0


@pytest.mark.parametrize(
    "enable_distributed_scheduling",
    [True, False],
)
def test_and_or_counts_match_when_all_arrays_update_together(
    enable_distributed_scheduling: bool, ray_multinode_cluster
) -> None:  # noqa: F811
    head_node_id, worker_node_ids = get_node_ids(ray_multinode_cluster)

    head_ref = head_script.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=head_node_id,
            soft=False,
        ),
    ).remote(enable_distributed_scheduling)
    wait_for_head_node()
    port = pick_free_port()

    worker_refs = [
        bridge_script_all_arrays.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=worker_node_ids[rank],
                soft=False,
            ),
        ).remote(rank=rank, port=port)
        for rank in range(2)
    ]

    results = ray.get([head_ref] + worker_refs)
    or_count, and_count = results[0]
    assert or_count == and_count
