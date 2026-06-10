# TODO is this really something we want to test? At the end of the day its a python feature... so maybe it should just be 
# shown as an example in the docs. 
import ray
import numpy as np

from deisa.ray.bridge import Bridge
from deisa.ray.comm import NoOpComm
from tests.utils import ray_multinode_cluster, wait_for_head_node  # noqa: F401
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

SHARED_SUM = 0


@ray.remote(max_retries=0)
def head_script() -> None:
    """The head node checks that the values are correct"""
    from deisa.ray.window_handler import Deisa

    global SHARED_SUM
    SHARED_SUM = 0

    d = Deisa()

    @d.register("array")
    def simulation_callback1(array):
        global SHARED_SUM
        x = array[0].sum().compute()
        assert x == 3
        SHARED_SUM = x

    @d.register("array")
    def simulation_callback2(array):
        x = array[0].sum().compute()
        assert x == SHARED_SUM

    d.execute_callbacks()


@ray.remote(max_retries=0)
def bridge_script(rank: int) -> str:
    arrays_md = {
        "array": {
            "global_shape": (1, 2),
            "chunk_shape": (1, 1),
            "chunk_position": (0, rank),
        }
    }

    bridge = Bridge(
        arrays_metadata=arrays_md,
        comm=NoOpComm(rank, 2),
    )  # type:ignore

    bridge.send(
        array_name="array",
        chunk=np.array([[rank + 1]], dtype=np.int64),
        timestep=0,
    )
    bridge.close(timestep=1)

    return bridge.node_id


def test_multiple_callbacks(ray_multinode_cluster) -> None:  # noqa: F811
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
    ).remote()
    wait_for_head_node()

    ray.get(
        bridge_script.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=worker_node_ids[1],
                soft=False,
            ),
        ).remote(1)
    )
    ray.get(
        bridge_script.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=worker_node_ids[0],
                soft=False,
            ),
        ).remote(0)
    )

    ray.get(head_ref)
