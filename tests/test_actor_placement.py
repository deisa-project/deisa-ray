import os
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from tests.stubs import StubSchedulingActor
from deisa.ray.bridge import Bridge
from tests.comm_utils import NoOpComm
from ray.util.state import list_actors
from deisa.ray.types import DeisaArray
from tests.utils import (
    pick_free_port,
    start_ray_multinode_cluster,
    wait_for_head_node,
)
from deisa.ray.utils import DEISA_HEAD_ACTOR_NAME, DEISA_NAMESPACE


def test_ray_multinode_clusters_can_start_in_parallel():
    clusters = [
        start_ray_multinode_cluster(head_node_gcs_server_port=pick_free_port()),
        start_ray_multinode_cluster(head_node_gcs_server_port=pick_free_port()),
    ]

    try:
        addresses = [cluster.address for cluster in clusters]
        assert len(set(addresses)) == len(addresses)
    finally:
        for cluster in clusters:
            cluster.shutdown()


@ray.remote
def head_script(enable_distributed_scheduling: bool = False) -> None:
    """The head node checks that the values are correct"""
    from deisa.ray.window_handler import Deisa

    os.environ["DEISA_DISTRIBUTED_SCHEDULING"] = "1" if enable_distributed_scheduling else "0"

    d = Deisa()

    @d.register("array")
    def simulation_callback(array: list[DeisaArray]):
        return True


@ray.remote
def bridge_script(rank):
    arrays_md = {
        "array": {
            "global_shape": (1, 2),
            "chunk_shape": (1, 1),
            "chunk_position": (0, rank),
        }
    }

    bridge = Bridge(
        arrays_metadata=arrays_md,
        comm=NoOpComm(0, 1),
        scheduling_actor_cls=StubSchedulingActor,
    )  # type:ignore

    return (bridge.node_id, f"sched-{bridge.node_id}")


def node_id_of_actor(name: str) -> str:
    for a in list_actors(
        address=ray.get_runtime_context().gcs_address,
        filters=[("state", "=", "ALIVE")],
    ):
        if a.get("name") == name and a.get("ray_namespace") == DEISA_NAMESPACE:
            # Newer Ray: address.node_id; older: may differ slightly
            nid = a.get("node_id")
            if nid:
                return nid
    raise RuntimeError(f"Actor {name} not found in namespace {DEISA_NAMESPACE}")


def test_actor_placement(ray_multinode_cluster):
    cluster = ray_multinode_cluster["cluster"]
    head_node_id = None
    worker_node_ids = []
    nodes = cluster.list_all_nodes()
    for node in nodes:
        if node.is_head():
            head_node_id = node.node_id
        else:
            worker_node_ids.append(node.node_id)

    # submit head script
    ray.get(
        head_script.options(
            max_retries=0, scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=head_node_id, soft=False)
        ).remote()
    )
    # wait for head actor to be up
    wait_for_head_node()

    # check that head actor is running in mock head node
    assert node_id_of_actor(DEISA_HEAD_ACTOR_NAME) == head_node_id

    nodes_to_actor = []
    for i, node in enumerate(worker_node_ids):
        nodes_to_actor.append(
            ray.get(
                bridge_script.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node, soft=False)
                ).remote(i)
            )
        )
    res = dict(nodes_to_actor)

    # assert node ids of bridge match worker ids
    assert list(res.keys()) == worker_node_ids

    actor_ids = []
    for name in res.values():
        actor_ids.append(node_id_of_actor(name))
    assert set(actor_ids) == set(worker_node_ids)
