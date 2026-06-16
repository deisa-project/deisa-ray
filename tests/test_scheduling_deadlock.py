import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from deisa.ray.head_node import HeadNodeActor
from deisa.ray.scheduling_actor import SchedulingActor
from deisa.ray.types import ScheduledByOtherActor
from deisa.ray.utils import get_head_actor_options
from tests.utils import ray_multinode_cluster  # noqa: F401


def test_cross_actor_graph_no_deadlock(ray_multinode_cluster) -> None:
    """
    Validate that cyclic cross-actor scheduling resolves without deadlock.

    Parameters
    ----------
    ray_multinode_cluster : dict[str, Any]
        Cluster fixture with node IDs and a connected driver.
    """
    worker_node_ids = []
    for node in ray_multinode_cluster["cluster"].list_all_nodes():
        if not node.is_head():
            worker_node_ids.append(node.node_id)

    node_ids = {
        "node1": worker_node_ids[0],
        "node2": worker_node_ids[1],
    }

    head = HeadNodeActor.options(**get_head_actor_options()).remote()
    ray.get(head.ready.remote())

    a0 = SchedulingActor.options(
        name="a0",
        namespace="deisa_ray",
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_ids["node1"], soft=False),
    ).remote("a0")
    a1 = SchedulingActor.options(
        name="a1",
        namespace="deisa_ray",
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_ids["node2"], soft=False),
    ).remote("a1")

    ray.get([a0.ready.remote(), a1.ready.remote()])

    graph_id = 123

    seed_k0 = ray.put(0)
    seed_k1 = ray.put(1)

    graph_a0 = {"k1": ScheduledByOtherActor("a1")}
    graph_a1 = {"k0": ScheduledByOtherActor("a0")}

    a0.schedule_graph.remote(graph_id, graph_a0, initial_refs={"k0": seed_k0})
    a1.schedule_graph.remote(graph_id, graph_a1, initial_refs={"k1": seed_k1})

    # equivalent to the last calls by "deisa_ray_get"
    res_k0 = a1.get_value.remote(graph_id, "k0")
    res_k1 = a0.get_value.remote(graph_id, "k1")

    vals = ray.get([res_k0, res_k1])
    assert vals == [0, 1]
