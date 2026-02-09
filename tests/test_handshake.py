import time
import dask.array as da
import pytest
import ray
import numpy as np
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.cluster_utils import Cluster
from deisa.ray.types import DeisaArray

@pytest.fixture
def ray_multinode_cluster():

    cluster = Cluster(
        initialize_head=True,
        connect=False,
        head_node_args={"num_cpus": 1},
    )
    cluster.add_node(num_cpus=1)
    cluster.add_node(num_cpus=1)
    cluster.add_node(num_cpus=1)
    cluster.add_node(num_cpus=1)

    # Connect driver to this cluster (IMPORTANT)
    ray.init(
        address=cluster.address,
        include_dashboard=False,
        log_to_driver=True,
        ignore_reinit_error=True,
    )

    yield {
        "cluster": cluster,
        "address": cluster.address,
    }

    ray.shutdown()
    cluster.shutdown()

NAMESPACE = "deisa_ray"

@pytest.mark.parametrize("sleep_t", [30])
def test_sim_start_first_and_analytics_after_x_secs(ray_multinode_cluster, sleep_t):
    cluster = ray_multinode_cluster["cluster"]
    head_node_id = None
    worker_nodes = []
    alive_nodes = 5
    while True:
        alive = [n for n in ray.nodes() if n["Alive"]]
        if len(alive) == alive_nodes:
            break

    for node in cluster.list_all_nodes():
        if node.is_head():
            head_node_id = node.node_id
        else:
            worker_nodes.append(node.node_id)

    @ray.remote(
        max_retries=0,
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=head_node_id, soft=False),
    )
    def head_script() -> bool:
        """The head node checks that the values are correct"""
        from deisa.ray.types import WindowSpec
        from deisa.ray.window_handler import Deisa

        d = Deisa(n_sim_nodes=4)

        def simulation_callback(array: list[DeisaArray]):
            pass

        d.register_callback(
            simulation_callback,
            [WindowSpec("array")],
        )
        d.execute_callbacks()
        return True

    # test that client creation resilient to head actor taking a long time to start
    @ray.remote
    def start_sim(rank, chunk_pos):
        from deisa.ray.utils import get_system_metadata
        from deisa.ray.bridge import Bridge

        arrays_md = {
            "array": {
                "chunk_shape": (1, 1),
                "nb_chunks_per_dim": (2, 2),
                "nb_chunks_of_node": 1,
                "dtype": np.int32,
                "chunk_position": chunk_pos,
            }
        }
        sys_md = get_system_metadata()
        try:
            b = Bridge(
                bridge_id=rank,
                arrays_metadata=arrays_md,
                system_metadata=sys_md,
                _node_id=None,
        )  # type:ignore
            b.close(timestep=0)
        except Exception:
            raise


        return b.node_id, f"sched-{b.node_id}"

    # submit sim first
    ref_sim = []
    for i, w_nid in enumerate(worker_nodes):
        ref_sim.append(
            start_sim
            .options(
                scheduling_strategy = NodeAffinitySchedulingStrategy(node_id=w_nid, soft=False)
            )
            .remote(
                i, (i // 2, i%2))
        )

    # submit analytics after sleep_t seconds
    time.sleep(sleep_t)
    ref_analytics = head_script.remote()

    sim_res = ray.get(ref_sim)
    for i, (n_id, _) in enumerate(sim_res):
        assert n_id == worker_nodes[i]
    assert ray.get(ref_analytics) == True


@pytest.mark.parametrize("sleep_t", [30])
def test_analytics_start_first_and_sim_after_x_secs(ray_multinode_cluster, sleep_t):
    cluster = ray_multinode_cluster["cluster"]
    head_node_id = None
    worker_nodes = []
    alive_nodes = 5
    while True:
        alive = [n for n in ray.nodes() if n["Alive"]]
        if len(alive) == alive_nodes:
            break

    for node in cluster.list_all_nodes():
        if node.is_head():
            head_node_id = node.node_id
        else:
            worker_nodes.append(node.node_id)

    @ray.remote(
        max_retries=0,
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=head_node_id, soft=False),
    )
    def head_script() -> bool:
        """The head node checks that the values are correct"""
        from deisa.ray.types import WindowSpec
        from deisa.ray.window_handler import Deisa

        d = Deisa(n_sim_nodes=4)

        def simulation_callback(array: list[DeisaArray]):
            pass

        d.register_callback(
            simulation_callback,
            [WindowSpec("array")],
        )
        d.execute_callbacks()
        return True

    # test that client creation resilient to head actor taking a long time to start
    @ray.remote
    def start_sim(rank, chunk_pos):
        from deisa.ray.utils import get_system_metadata
        from deisa.ray.bridge import Bridge

        arrays_md = {
            "array": {
                "chunk_shape": (1, 1),
                "nb_chunks_per_dim": (2, 2),
                "nb_chunks_of_node": 1,
                "dtype": np.int32,
                "chunk_position": chunk_pos,
            }
        }
        sys_md = get_system_metadata()
        try:
            b = Bridge(
                bridge_id=rank,
                arrays_metadata=arrays_md,
                system_metadata=sys_md,
                _node_id=None,
            )  # type:ignore
            b.close(timestep=0)
        except Exception:
            raise


        return b.node_id, f"sched-{b.node_id}"


    # submit analytics first
    ref_analytics = head_script.remote()
    time.sleep(sleep_t)
    # submit sim after sleep_t seconds
    ref_sim = []
    for i, w_nid in enumerate(worker_nodes):
        ref_sim.append(
            start_sim
            .options(
                scheduling_strategy = NodeAffinitySchedulingStrategy(node_id=w_nid, soft=False)
            )
            .remote(
                i, (i // 2, i%2))
        )

    sim_res = ray.get(ref_sim)
    for i, (n_id, _) in enumerate(sim_res):
        assert n_id == worker_nodes[i]
    assert ray.get(ref_analytics) == True

def test_analytics_raise_if_not_enough_actors_connect(ray_multinode_cluster):
    with pytest.raises(RuntimeError):
        cluster = ray_multinode_cluster["cluster"]
        head_node_id = None
        worker_nodes = []
        alive_nodes = 5
        while True:
            alive = [n for n in ray.nodes() if n["Alive"]]
            if len(alive) == alive_nodes:
                break

        for node in cluster.list_all_nodes():
            if node.is_head():
                head_node_id = node.node_id
            else:
                worker_nodes.append(node.node_id)


        @ray.remote(
            max_retries=0,
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=head_node_id, soft=False),
        )
        def head_script() -> bool:
            """The head node checks that the values are correct"""
            from deisa.ray.types import WindowSpec
            from deisa.ray.window_handler import Deisa

            d = Deisa(n_sim_nodes=4, _timeout_s=10)

            def simulation_callback(array: list[DeisaArray]):
                pass

            d.register_callback(
                simulation_callback,
                [WindowSpec("array")],
            )
            d.execute_callbacks()
            return True


        # test that client creation resilient to head actor taking a long time to start
        @ray.remote
        def start_sim(rank, chunk_pos):
            from deisa.ray.utils import get_system_metadata
            from deisa.ray.bridge import Bridge

            arrays_md = {
                "array": {
                    "chunk_shape": (1, 1),
                    "nb_chunks_per_dim": (2, 2),
                    "nb_chunks_of_node": 1,
                    "dtype": np.int32,
                    "chunk_position": chunk_pos,
                }
            }
            sys_md = get_system_metadata()
            try:
                b = Bridge(
                    bridge_id=rank,
                    arrays_metadata=arrays_md,
                    system_metadata=sys_md,
                    _node_id=None,
                )  # type:ignore
                b.close(timestep=0)
            except Exception:
                raise

            return b.node_id, f"sched-{b.node_id}"


        # submit analytics first
        ref_analytics = head_script.remote()
        # submit sim after sleep_t seconds
        ref_sim = []
        # NOTE: only 3 nodes (so only 3 actors will connect)
        for i, w_nid in enumerate(worker_nodes[:3]):
            ref_sim.append(
                start_sim
                .options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=w_nid, soft=False)
                )
                .remote(
                    i, (i // 2, i % 2))
            )

        sim_res = ray.get(ref_sim)
        for i, (n_id, _) in enumerate(sim_res):
            assert n_id == worker_nodes[i]
        assert ray.get(ref_analytics) == True


# TODO window handler raises runtime error but also on bridge the Bridge becomes null
# we need one path of decision.
def test_analytics_raise_if_too_many_actors_connect(ray_multinode_cluster):
    with pytest.raises(RuntimeError):
        cluster = ray_multinode_cluster["cluster"]
        head_node_id = None
        worker_nodes = []
        alive_nodes = 5
        while True:
            alive = [n for n in ray.nodes() if n["Alive"]]
            if len(alive) == alive_nodes:
                break

        for node in cluster.list_all_nodes():
            if node.is_head():
                head_node_id = node.node_id
            else:
                worker_nodes.append(node.node_id)

        @ray.remote(
            max_retries=0,
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=head_node_id, soft=False),
        )
        def head_script() -> bool:
            """The head node checks that the values are correct"""
            from deisa.ray.types import WindowSpec
            from deisa.ray.window_handler import Deisa

            # NOTE!! changed to one node
            d = Deisa(n_sim_nodes=1)

            def simulation_callback(array: list[DeisaArray]):
                pass

            d.register_callback(
                simulation_callback,
                [WindowSpec("array")],
            )
            d.execute_callbacks()
            return True

        # test that client creation resilient to head actor taking a long time to start
        @ray.remote
        def start_sim(rank, chunk_pos):
            from deisa.ray.utils import get_system_metadata
            from deisa.ray.bridge import Bridge

            arrays_md = {
                "array": {
                    "chunk_shape": (1, 1),
                    # TODO: if 2,2 as before, its stuck at deadlock bc it waits until appropriate num of chunks arrive
                    "nb_chunks_per_dim": (2, 2),
                    "nb_chunks_of_node": 1,
                    "dtype": np.int32,
                    "chunk_position": chunk_pos,
                }
            }
            sys_md = get_system_metadata()
            try:
                b = Bridge(
                    bridge_id=rank,
                    arrays_metadata=arrays_md,
                    system_metadata=sys_md,
                    _node_id=None,
                )  # type:ignore
                b.close(timestep=0)
            except Exception:
                raise


            return b.node_id, f"sched-{b.node_id}"


        # submit analytics first
        ref_analytics = head_script.remote()
        # submit sim after sleep_t seconds
        ref_sim = []
        for i, w_nid in enumerate(worker_nodes):
            ref_sim.append(
                start_sim
                .options(
                    scheduling_strategy = NodeAffinitySchedulingStrategy(node_id=w_nid, soft=False)
                )
                .remote(
                    i, (i // 2, i%2))
            )

        sim_res = ray.get(ref_sim)
        for i, (n_id, _) in enumerate(sim_res):
            assert n_id == worker_nodes[i]
        assert ray.get(ref_analytics) == True

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
