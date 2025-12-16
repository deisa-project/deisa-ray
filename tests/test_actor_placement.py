import pytest
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from tests.stubs import StubSchedulingActor
from deisa.ray.bridge import Bridge
from ray.util.state import list_actors
import dask.array as da
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


def test_fake_cluster(ray_multinode_cluster):
    cluster = ray_multinode_cluster["cluster"]
    ids = ray_multinode_cluster["ids"]
    nodes = cluster.list_all_nodes()
    for node in nodes:
        print(f"Node id: {node.node_id}, is_head: {node.is_head()}")
    assert len(nodes) == len(ids)
    assert ray.is_initialized()
    assert cluster.gcs_address is not None
    # Check that overridden node IDs are present
    live_ids = {n.node_id for n in nodes}
    assert set(ids.values()) == live_ids


NAMESPACE = "deisa_ray"


def actor_node_id_by_name(name: str, namespace: str = NAMESPACE) -> str:
    for a in list_actors(filters=[("state", "=", "ALIVE")]):
        if a.get("name") == name and a.get("ray_namespace") == namespace:
            # Newer Ray: address.node_id; older: may differ slightly
            nid = a.get("node_id")
            if nid:
                return nid
    raise RuntimeError(f"Actor {name} not found in namespace {namespace}")


@pytest.mark.parametrize(
        "enable_distributed_scheduling", 
        [True, False]
)
def test_actor_placement(enable_distributed_scheduling, ray_multinode_cluster):
    ids = ray_multinode_cluster["ids"]
    head_node_id, worker_node_id = ids["head"], ids["node1"]

    # start analytics first (this ensures that head actor is created. We have an explicit wait)
    # make sure that it runs on mock head node
    @ray.remote(
        max_retries=0,
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=head_node_id, soft=False),
    )
    def head_script(enable_distributed_scheduling) -> None:
        """The head node checks that the values are correct"""
        from deisa.ray.window_handler import Deisa
        from deisa.ray.types import WindowArrayDefinition

        import deisa.ray as deisa

        deisa.config.enable_experimental_distributed_scheduling(enable_distributed_scheduling)

        d = Deisa()

        def simulation_callback(array: da.Array, timestep: int):
            return True

        d.register_callback(
            simulation_callback,
            [WindowArrayDefinition("array")],
            max_iterations=0,
        )
        d.execute_callbacks()

    # submit head script (analogous to submitting analytics to head node)
    ray.get(head_script.remote(enable_distributed_scheduling))
    # just a ray.get_actor() wrapped in while loop
    wait_for_head_node()

    # check that head actor is running in mock head node
    assert actor_node_id_by_name("simulation_head") == head_node_id

    # start client in worker node (this will create one scheduling actor)
    # we need to check that both the scheduling actor and client have node_id that is the same as
    # the worker node id
    @ray.remote(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=worker_node_id, soft=False),
    )
    def make_client_and_return_ids():
        from deisa.ray.utils import get_system_metadata

        sys_md = get_system_metadata()
        c = Bridge(
            id=0, arrays_metadata={}, system_metadata=sys_md, _node_id=None, scheduling_actor_cls=StubSchedulingActor
        )  # type:ignore

        return (c.node_id, f"sched-{c.node_id}")

    client_node_id, sched_name = ray.get(make_client_and_return_ids.remote())
    assert client_node_id == worker_node_id
    assert actor_node_id_by_name(sched_name) == worker_node_id


# @pytest.mark.parametrize("sleep_t", [0, 60, 120])
# def test_analytics_late_start(ray_multinode_cluster, sleep_t):
#     ids = ray_multinode_cluster["ids"]
#     head_node_id, worker_node_id = ids["head"], ids["node1"]
#
#     @ray.remote(
#         max_retries=0,
#         scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=head_node_id, soft=False),
#     )
#     def head_script(enable_distributed_scheduling) -> None:
#         """The head node checks that the values are correct"""
#         from deisa.ray.window_api import ArrayDefinition, run_simulation
#
#         time.sleep(sleep_t)
#
#         def simulation_callback(array: da.Array, timestep: int):
#             return True
#
#         run_simulation(
#             simulation_callback,
#             [ArrayDefinition("array")],
#             max_iterations=0,
#         )
#
#     head_script.remote()
#
#     # test that client creation resilient to head actor taking a long time to start
#     @ray.remote(
#         scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=worker_node_id, soft=False),
#     )
#     def make_client_and_return_ids():
#         c = Bridge(_node_id=None)  # type:ignore
#         return (c.node_id, f"sched-{c.node_id}")
#
#     # this should be blocking because we want the sim code to wait
#     client_node_id, sched_name = ray.get(make_client_and_return_ids.remote())
#
#     # check placement
#     assert client_node_id == worker_node_id
#     assert actor_node_id_by_name(sched_name) == worker_node_id
#     assert actor_node_id_by_name("simulation_head") == head_node_id
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
