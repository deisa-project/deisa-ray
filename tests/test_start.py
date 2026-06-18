import time
import pytest
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from deisa.ray.types import DeisaArray
import torch.distributed as dist
from tests.utils import pick_free_port

# TODO check that all errors types that can be raised when the bridge is not properly initialized are correctly caught raised
# for now we catch DistStoreError and DistNetworkError but should check that the other errors also happen correctly.
# needed tests:
# 1) RuntimeError if node actor cannot be created after N retries.
# 2) Value Error if comm is None
# 3) Type Error if comm is not of type ICommunicator
# 4) No rank0 raises an error

DIST_TIMEOUT_ERRORS = tuple(
    err
    for err in (
        getattr(dist, "DistStoreError", None),
        getattr(dist, "DistNetworkError", None),
    )
    if err is not None
)


@ray.remote(max_retries=0)
def head_script() -> bool:
    """The head node checks that the values are correct"""
    from deisa.ray.window_handler import Deisa

    d = Deisa()

    @d.register("array")
    def simulation_callback(array: list[DeisaArray]):
        pass

    d.execute_callbacks()
    return True


@ray.remote(num_cpus=0, max_retries=0)
def bridge_script(*, rank: int, port: int):
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
        timeout_s=1,
    )
    b = Bridge(
        arrays_metadata=arrays_md,
        comm=comm,
    )  # type:ignore


@pytest.mark.parametrize("sleep_t", [5])
def test_sim_start_first_and_analytics_can_start_after_x_secs(ray_multinode_cluster, sleep_t):
    gloo_port = pick_free_port()
    cluster = ray_multinode_cluster["cluster"]
    head_node_id = None
    worker_node_ids = []
    for node in cluster.list_all_nodes():
        if node.is_head():
            head_node_id = node.node_id
        else:
            worker_node_ids.append(node.node_id)

    # submit sim first
    worker_refs = [
        bridge_script.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=worker_node_ids[rank],
                soft=False,
            ),
        ).remote(rank=rank, port=gloo_port)
        for rank in range(2)
    ]

    # submit analytics after sleep_t seconds
    time.sleep(sleep_t)
    head_ref = head_script.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=head_node_id,
            soft=False,
        )
    ).remote()

    ray.get([head_ref] + worker_refs)


@pytest.mark.parametrize("sleep_t", [5])
def test_analytics_start_first_and_sim_can_start_after_x_secs(ray_multinode_cluster, sleep_t):
    gloo_port = pick_free_port()
    cluster = ray_multinode_cluster["cluster"]
    head_node_id = None
    worker_node_ids = []
    for node in cluster.list_all_nodes():
        if node.is_head():
            head_node_id = node.node_id
        else:
            worker_node_ids.append(node.node_id)

    # submit analytics first
    head_ref = head_script.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=head_node_id,
            soft=False,
        ),
    ).remote()
    time.sleep(sleep_t)

    # submit sim after sleep_t seconds
    worker_refs = [
        bridge_script.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=worker_node_ids[rank],
                soft=False,
            ),
        ).remote(rank=rank, port=gloo_port)
        for rank in range(2)
    ]

    ray.get([head_ref] + worker_refs)


def test_sim_raise_if_not_enough_bridges_connect(ray_multinode_cluster):
    with pytest.raises(DIST_TIMEOUT_ERRORS):
        gloo_port = pick_free_port()
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

        # NOTE: we only have 1 node (so only 1 Bridges will be created)
        worker_refs = [
            bridge_script.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=worker_node_ids[0],
                    soft=False,
                ),
            ).remote(rank=0, port=gloo_port)
        ]

        ray.get([head_ref] + worker_refs)
