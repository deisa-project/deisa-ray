import os

import numpy as np
import pytest
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from deisa.ray.types import DeisaArray
from tests.utils import pick_free_port, ray_multinode_cluster, wait_for_head_node  # noqa: F401

NB_ITERATIONS = 5


@ray.remote(max_retries=0)
def head_script(enable_distributed_scheduling) -> None:
    """The head node checks that the values are correct"""
    from deisa.ray.window_handler import Deisa

    os.environ["DEISA_DISTRIBUTED_SCHEDULING"] = "1" if enable_distributed_scheduling else "0"

    d = Deisa()

    @d.register("array")
    def simulation_callback(array: list[DeisaArray]):
        x = array[0].mean().compute()
        assert x == (3 * array[0].t) / 2
    d.execute_callbacks()


@ray.remote(num_cpus=0, max_retries=0)
def bridge_script(*, rank: int, port: int) -> str:
    import io
    import logging

    from deisa.ray.bridge import Bridge
    from tests.comm_utils import init_gloo_comm

    arrays_md = {
        "array": {
            "global_shape": (1, 2),
            "chunk_shape": (1, 1),
            "chunk_position": (0, rank),
        }
    }

    comm = init_gloo_comm(
        2,
        rank,
        "127.0.0.1",
        port,
    )
    bridge = Bridge(arrays_metadata=arrays_md, comm=comm, _node_id=f"node_{rank}")

    array = (rank + 1) * np.ones((1, 1), dtype=np.int32)
    for i in range(NB_ITERATIONS):
        bridge.send(array_name="array", chunk=i * array, timestep=i)

    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    logger = logging.getLogger("deisa.ray.bridge")
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    bridge.close(timestep=NB_ITERATIONS)
    logger.removeHandler(handler)
    logger.setLevel(previous_level)

    return log_stream.getvalue()


@pytest.mark.parametrize("enable_distributed_scheduling", (True, False))
def test_deisa_ray(enable_distributed_scheduling: bool, ray_multinode_cluster) -> None:  # noqa: F811
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

    results = ray.get([head_ref] + worker_refs)
    assert any(f"Bridge 0 closed at timestep {NB_ITERATIONS}" in result for result in results[1:])

    # Check that the right number of scheduling actors were created
    simulation_head = ray.get_actor("simulation_head", namespace="deisa_ray")
    assert len(ray.get(simulation_head.list_scheduling_actors.remote())) == 2
