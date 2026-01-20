# import pytest
# import ray
# from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
# from tests.stubs import StubSchedulingActor
# from deisa.ray.bridge import Bridge
# from ray.util.state import list_actors
# import dask.array as da
# from ray.cluster_utils import Cluster
# from tests.utils import wait_for_head_node
# import time
# import numpy as np
#
# @pytest.fixture
# def ray_multinode_cluster():
#     cluster_node_ids = {
#         "head": "f64704987dec54e6c20445dc6a063ad34de1cd777d5c7e0779d1100a",
#         "node1": "f64704987dec54e6c20445dc6a063ad34de1cd777d5c7e0779d1100b",
#         "node2": "f64704987dec54e6c20445dc6a063ad34de1cd777d5c7e0779d1100c",
#     }
#
#     cluster = Cluster(
#         initialize_head=True,
#         connect=False,
#         head_node_args={
#             "num_cpus": 1,
#             "env_vars": {"RAY_OVERRIDE_NODE_ID_FOR_TESTING": cluster_node_ids["head"]},
#         },
#     )
#
#     cluster.add_node(num_cpus=1, env_vars={"RAY_OVERRIDE_NODE_ID_FOR_TESTING": cluster_node_ids["node1"]})
#
#     # Connect driver to this cluster (IMPORTANT)
#     ray.init(
#         address=cluster.address,
#         include_dashboard=False,
#         log_to_driver=True,
#         ignore_reinit_error=True,
#     )
#
#     yield {
#         "cluster": cluster,
#         "ids": cluster_node_ids,
#         "address": cluster.address,
#     }
#
#     ray.shutdown()
#     cluster.shutdown()
#
#
# def test_set_from_analytics(ray_multinode_cluster):
#     ids = ray_multinode_cluster["ids"]
#     head_node_id  = ids["head"]
#     worker_node_ids = [ids["node1"], ids["node2"]]
#
#     @ray.remote(
#         max_retries=0,
#         scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=head_node_id, soft=False),
#     )
#     def head_script(enable_distributed_scheduling) -> None:
#         """The head node checks that the values are correct"""
#         from deisa.ray.window_api import Deisa
#         from deisa.ray.types import WindowSpec
#
#         import deisa.ray as deisa
#
#    deisa.config.enable_experimental_distributed_scheduling(enable_distributed_scheduling)
#
#    d = Deisa()
#         def cb(array, timestep):
#             deisa.set(key = "foo", value = True)
#         d.register_callback(cb, [WindowSpec("array")], max_iterations=1)
#         d.execute_callbacks()
#
#     @ray.remote
#     def make_bridge(i):
#         from deisa.ray.utils import get_system_metadata
#         sys_md = get_system_metadata()
#         c = Bridge(id = i, arrays_metadata= {}, system_metadata= sys_md)  # type:ignore
#         chunk = i * np.ones((2,))
#         c.send(array_name="array", chunk=chunk, timestep= 0)
#         feedback = c.get(name = "foo")
#         return (c.node_id, f"sched-{c.node_id}", feedback)
#
#     # create 2 bridges (will create two node actors)
#     refs = []
#     for i,w_id in enumerate(worker_node_ids):
#         refs.append(make_bridge.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=w_id, soft=False)).remote(i))
#
#     # submit Deisa
#     head_script.remote()
#
#     # retrieve res of refs
#     c1, n1, f1 = ray.get(refs[0])
#     c2, n2, f2 = ray.get(refs[1])
#
#     assert f1 is True
#     assert f2 is True
#
#
#
