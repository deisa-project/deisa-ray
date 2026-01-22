import time
import dask.array as da
import pytest
import ray
import numpy as np
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from deisa.ray.bridge import Bridge
from ray.util.state import list_actors
from deisa.ray.types import DeisaArray
from ray.cluster_utils import Cluster
from tests.utils import wait_for_head_node

@pytest.fixture
def ray_multinode_cluster():
    cluster_node_ids = {
        "head": "f64704987dec54e6c20445dc6a063ad34de1cd777d5c7e0779d1100a",
        "node1": "f64704987dec54e6c20445dc6a063ad34de1cd777d5c7e0779d1100b",
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

    # Connect driver to this cluster (IMPORTANT)
    ray.init(
        address=cluster.address,
        include_dashboard=False,
        log_to_driver=True,
        ignore_reinit_error=True,
    )

    yield {
        "cluster": cluster,
        "ids": cluster_node_ids,
        "address": cluster.address,
    }

    ray.shutdown()
    cluster.shutdown()

NAMESPACE = "deisa_ray"

def actor_node_id_by_name(name: str, namespace: str = NAMESPACE) -> str:
    for a in list_actors(filters=[("state", "=", "ALIVE")]):
        if a.get("name") == name and a.get("ray_namespace") == namespace:
            # Newer Ray: address.node_id; older: may differ slightly
            nid = a.get("node_id")
            if nid:
                return nid
    raise RuntimeError(f"Actor {name} not found in namespace {namespace}")

@pytest.mark.parametrize("sleep_t", [0, 60, 120])
def test_sim_start_first_and_analytics_after_x_secs(ray_multinode_cluster, sleep_t):
    ids = ray_multinode_cluster["ids"]
    head_node_id, worker_node_id = ids["head"], ids["node1"]

    @ray.remote(
        max_retries=0,
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=head_node_id, soft=False),
    )
    def head_script() -> None:
        """The head node checks that the values are correct"""
        from deisa.ray.types import WindowSpec
        from deisa.ray.window_handler import Deisa

        deisa = Deisa(1)
        def simulation_callback(array: da.Array, timestep: int):
            pass

        deisa.register_callback(simulation_callback, [WindowSpec("array")])
        deisa.execute_callbacks()


    # test that client creation resilient to head actor taking a long time to start
    @ray.remote(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=worker_node_id, soft=False),
    )
    def start_sim():
        from deisa.ray.utils import get_system_metadata

        arrays_md = {
            "array": {
                "chunk_shape": (1, 1),
                "nb_chunks_per_dim": (1, 1),
                "nb_chunks_of_node": 1,
                "dtype": np.int32,
                "chunk_position": (0, 0),
            }
        }
        sys_md = get_system_metadata()
        c = Bridge(
            id=0,
            arrays_metadata=arrays_md,
            system_metadata=sys_md,
            _node_id=None,
        )  # type:ignore

        return (c.node_id, f"sched-{c.node_id}")

    # submit sim first so it starts
    ref_sim = start_sim.remote()

    # submit analytics after sleep_t seconds
    time.sleep(sleep_t)
    ref_analytics = head_script.remote()

    bridge_node_id, node_actor_name = ray.get(ref_sim)



    # check placement
    assert bridge_node_id == worker_node_id
    assert actor_node_id_by_name(node_actor_name) == worker_node_id
    assert actor_node_id_by_name("simulation_head") == head_node_id

@pytest.mark.parametrize("sleep_t", [0, 60, 120])
def test_analytics_start_first_and_sim_after_x_secs(ray_multinode_cluster, sleep_t):
    ids = ray_multinode_cluster["ids"]
    head_node_id, worker_node_id = ids["head"], ids["node1"]

    @ray.remote(
        max_retries=0,
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=head_node_id, soft=False),
    )
    def head_script() -> None:
        """The head node checks that the values are correct"""
        from deisa.ray.types import WindowSpec
        from deisa.ray.window_handler import Deisa

        deisa = Deisa(1)
        def simulation_callback(array: da.Array, timestep: int):
            pass

        deisa.register_callback(simulation_callback, [WindowSpec("array")])
        deisa.execute_callbacks()


    # test that client creation resilient to head actor taking a long time to start
    @ray.remote(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=worker_node_id, soft=False),
    )
    def start_sim():
        from deisa.ray.utils import get_system_metadata

        arrays_md = {
            "array": {
                "chunk_shape": (1, 1),
                "nb_chunks_per_dim": (1, 1),
                "nb_chunks_of_node": 1,
                "dtype": np.int32,
                "chunk_position": (0, 0),
            }
        }
        sys_md = get_system_metadata()
        c = Bridge(
            id=0,
            arrays_metadata=arrays_md,
            system_metadata=sys_md,
            _node_id=None,
        )  # type:ignore

        return (c.node_id, f"sched-{c.node_id}")


    # submit analytics after sleep_t seconds
    ref_analytics = head_script.remote()

    time.sleep(sleep_t)

    # submit sim first so it starts
    ref_sim = start_sim.remote()
    bridge_node_id, node_actor_name = ray.get(ref_sim)

    # check placement
    assert bridge_node_id == worker_node_id
    assert actor_node_id_by_name(node_actor_name) == worker_node_id
    assert actor_node_id_by_name("simulation_head") == head_node_id
#
#
# # TODO use more specific timeoutError
# def test_sim_exits_if_analytics_dont_start(ray_multinode_cluster):
#     ids = ray_multinode_cluster["ids"]
#     worker_node_id = ids["node1"]
#
#     # test that client creation resilient to head actor taking a long time to start
#     @ray.remote(
#         scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=worker_node_id, soft=False),
#     )
#     def make_client_and_return_ids():
#         c = Bridge(_node_id=None, _init_retries=1)  # type:ignore
#         return (c.node_id, f"sched-{c.node_id}")
#
#     with pytest.raises(Exception):
#         ray.get(make_client_and_return_ids.remote())
