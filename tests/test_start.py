import time
import pytest
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from deisa.ray.types import DeisaArray
import torch.distributed as dist
from tests.utils import pick_free_port, ray_multinode_cluster

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

@ray.remote
def head_script() -> bool:
    """The head node checks that the values are correct"""
    from deisa.ray.window_handler import Deisa

    d = Deisa()

    @d.register("array")
    def simulation_callback(array: list[DeisaArray]):
        pass

    d.execute_callbacks()
    return True

@ray.remote
def bridge_script(rank, port):
    from deisa.ray.bridge import Bridge
    from tests.comm_utils import init_gloo_comm

    arrays_md = {
        "array": {
            "global_shape": (1, 2),
            "chunk_shape": (1, 1),
            "chunk_position": (0, rank),
        }
    }
    try:
        comm = init_gloo_comm(
            2,
            rank,
            "127.0.0.1",
            port,
            timeout_s=10,
        )
        b = Bridge(
            arrays_metadata=arrays_md,
            comm=comm,
        )  # type:ignore
    # TODO split exception into two cases: one for init_gloo_comm having some timeout, and another for Bridges
    # not joining within threshold time. 
    except Exception:
        raise

    return b.node_id, f"sched-{b.node_id}"

@pytest.mark.parametrize("sleep_t", [5])
def test_sim_start_first_and_analytics_can_start_after_x_secs(ray_multinode_cluster, sleep_t):
    gloo_port = pick_free_port()
    cluster = ray_multinode_cluster["cluster"]
    head_node_id = None
    worker_nodes = []
    for node in cluster.list_all_nodes():
        if node.is_head():
            head_node_id = node.node_id
        else:
            worker_nodes.append(node.node_id)

    # submit sim first
    ref_sim = []
    for i, w_nid in enumerate(worker_nodes):
        ref_sim.append(
            bridge_script.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=w_nid, soft=False)).remote(
                i, gloo_port
            )
        )

    # submit analytics after sleep_t seconds
    time.sleep(sleep_t)
    ref_analytics = head_script.options(
        max_retries=0,
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=head_node_id, soft=False),
    ).remote()

    sim_res = ray.get(ref_sim)
    for i, (n_id, _) in enumerate(sim_res):
        assert n_id == worker_nodes[i]
    assert ray.get(ref_analytics)


@pytest.mark.parametrize("sleep_t", [5])
def test_analytics_start_first_and_sim_can_start_after_x_secs(ray_multinode_cluster, sleep_t):
    gloo_port = pick_free_port()
    cluster = ray_multinode_cluster["cluster"]
    head_node_id = None
    worker_nodes = []
    for node in cluster.list_all_nodes():
        if node.is_head():
            head_node_id = node.node_id
        else:
            worker_nodes.append(node.node_id)

    # submit analytics first
    ref_analytics = head_script.options(
        max_retries=0,
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=head_node_id, soft=False),
    ).remote()
    time.sleep(sleep_t)
    # submit sim after sleep_t seconds
    ref_sim = []
    for i, w_nid in enumerate(worker_nodes):
        ref_sim.append(
            bridge_script.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=w_nid, soft=False)).remote(
                i, gloo_port
            )
        )

    sim_res = ray.get(ref_sim)
    for i, (n_id, _) in enumerate(sim_res):
        assert n_id == worker_nodes[i]
    assert ray.get(ref_analytics)


def test_sim_raise_if_not_enough_bridges_connect(ray_multinode_cluster):
    with pytest.raises(DIST_TIMEOUT_ERRORS):
        gloo_port = pick_free_port()
        cluster = ray_multinode_cluster["cluster"]
        head_node_id = None
        worker_nodes = []
        for node in cluster.list_all_nodes():
            if node.is_head():
                head_node_id = node.node_id
            else:
                worker_nodes.append(node.node_id)

        ref_analytics = head_script.options( 
            max_retries=0,
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=head_node_id, soft=False),
        ).remote()
        ref_sim = []
        # NOTE: we only have 1 node (so only 1 Bridges will be created)
        w_nid = worker_nodes[0]
        ref_sim.append(
                bridge_script.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=w_nid, soft=False)).remote(
                    0, gloo_port
                )
            )
        sim_res = ray.get(ref_sim)
        for i, (n_id, _) in enumerate(sim_res):
            assert n_id == worker_nodes[i]
        assert ray.get(ref_analytics)