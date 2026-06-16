import os

import numpy as np
import pytest
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from deisa.ray.types import DeisaArray
from tests.utils import pick_free_port, ray_multinode_cluster, wait_for_head_node  # noqa: F401

NB_ITERATIONS = 5


@ray.remote(max_retries=0)
def head_script(enable_distributed_scheduling: bool) -> None:
    from deisa.ray.window_handler import Deisa

    os.environ["DEISA_DISTRIBUTED_SCHEDULING"] = "1" if enable_distributed_scheduling else "0"

    d = Deisa()

    @d.register("array")
    def simulation_callback1(array: list[DeisaArray]) -> None:
        assert array[0].sum().compute() == 3 * array[0].t

    @d.register("array1")
    def simulation_callback2(array1: list[DeisaArray]) -> None:
        assert array1[0].sum().compute() == 3 * array1[0].t

    @d.register("array", "array1")
    def simulation_callback3(array: list[DeisaArray], array1: list[DeisaArray]) -> None:
        assert array[0].sum().compute() == 3 * array[0].t
        assert array1[0].sum().compute() == 3 * array1[0].t

    d.execute_callbacks()


@ray.remote(num_cpus=0, max_retries=0)
def bridge_script(*, rank: int, port: int) -> None:
    from deisa.ray.bridge import Bridge
    from tests.comm_utils import init_gloo_comm

    arrays_md = {
        name: {
            "global_shape": (1, 2),
            "chunk_shape": (1, 1),
            "chunk_position": (0, rank),
        }
        for name in ["array", "array1"]
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
        for array_name in arrays_md:
            bridge.send(array_name=array_name, chunk=timestep * array, timestep=timestep)

    bridge.close(timestep=NB_ITERATIONS)


@pytest.mark.parametrize("enable_distributed_scheduling", [True, False])
def test_multiple_callbacks(enable_distributed_scheduling: bool, ray_multinode_cluster) -> None:  # noqa: F811
    cluster = ray_multinode_cluster["cluster"]
    head_node_id = None
    worker_node_ids = []
    for node in cluster.list_all_nodes():
        if node.is_head():
            head_node_id = node.node_id
        else:
            worker_node_ids.append(node.node_id)

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

    ray.get([head_ref] + worker_refs)
