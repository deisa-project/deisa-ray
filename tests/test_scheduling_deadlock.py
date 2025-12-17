from typing import Any

import pytest
import ray
from ray.cluster_utils import Cluster
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from deisa.ray.head_node import HeadNodeActor
from deisa.ray.scheduling_actor import SchedulingActor
from deisa.ray.types import ScheduledByOtherActor
from deisa.ray.utils import get_head_actor_options


@pytest.fixture
def ray_three_node_cluster() -> dict[str, Any]:
    """
    Start a three-node Ray cluster for scheduling-actor tests.

    Returns
    -------
    dict[str, Any]
        Mapping containing the cluster handle and stable node IDs.
    """
    cluster_node_ids = {
        "head": "f64704987dec54e6c20445dc6a063ad34de1cd777d5c7e0779d1100a",
        "node1": "f64704987dec54e6c20445dc6a063ad34de1cd777d5c7e0779d1100b",
        "node2": "f64704987dec54e6c20445dc6a063ad34de1cd777d5c7e0779d1100c",
    }

    cluster = Cluster(
        initialize_head=True,
        connect=False,
        head_node_args={
            "num_cpus": 1,
            "env_vars": {"RAY_OVERRIDE_NODE_ID_FOR_TESTING": cluster_node_ids["head"]},
        },
    )

    cluster.add_node(num_cpus=1, env_vars={"RAY_OVERRIDE_NODE_ID_FOR_TESTING": cluster_node_ids["node1"]})
    cluster.add_node(num_cpus=1, env_vars={"RAY_OVERRIDE_NODE_ID_FOR_TESTING": cluster_node_ids["node2"]})

    ray.init(
        address=cluster.address,
        include_dashboard=False,
        log_to_driver=True,
        ignore_reinit_error=True,
    )

    yield {"cluster": cluster, "ids": cluster_node_ids}

    ray.shutdown()
    cluster.shutdown()


def test_cross_actor_graph_no_deadlock(ray_three_node_cluster: dict[str, Any]) -> None:
    """
    Validate that cyclic cross-actor scheduling resolves without deadlock.

    Parameters
    ----------
    ray_three_node_cluster : dict[str, Any]
        Cluster fixture with node IDs and a connected driver.
    """
    node_ids = ray_three_node_cluster["ids"]

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
