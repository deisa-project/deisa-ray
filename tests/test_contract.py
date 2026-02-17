import time
import pytest
import ray
import numpy as np
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.cluster_utils import Cluster
from deisa.ray.types import DeisaArray
from torch.distributed import DistStoreError
from tests.utils import pick_free_port


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
def test_sim_start_first_and_analytics_can_start_after_x_secs(ray_multinode_cluster, sleep_t):
    port = pick_free_port()
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

        d = Deisa()

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
    def start_sim(rank, chunk_pos, port):
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

        sys_md = {"world_size": 4, "master_address": "127.0.0.1", "master_port": port}
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
            start_sim.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=w_nid, soft=False)).remote(
                i, (i // 2, i % 2), port
            )
        )

    # submit analytics after sleep_t seconds
    time.sleep(sleep_t)
    ref_analytics = head_script.remote()

    sim_res = ray.get(ref_sim)
    for i, (n_id, _) in enumerate(sim_res):
        assert n_id == worker_nodes[i]
    assert ray.get(ref_analytics)


@pytest.mark.parametrize("sleep_t", [30])
def test_analytics_start_first_and_sim_can_start_after_x_secs(ray_multinode_cluster, sleep_t):
    port = pick_free_port()
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

        d = Deisa()

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
    def start_sim(rank, chunk_pos, port):
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

        sys_md = {"world_size": 4, "master_address": "127.0.0.1", "master_port": port}
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
            start_sim.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=w_nid, soft=False)).remote(
                i, (i // 2, i % 2), port
            )
        )

    sim_res = ray.get(ref_sim)
    for i, (n_id, _) in enumerate(sim_res):
        assert n_id == worker_nodes[i]
    assert ray.get(ref_analytics)


def test_sim_raise_if_not_enough_bridges_connect(ray_multinode_cluster):
    with pytest.raises(DistStoreError):
        port = pick_free_port()
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

            d = Deisa()

            def simulation_callback(array: list[DeisaArray]):
                pass

            d.register_callback(
                simulation_callback,
                [WindowSpec("array")],
            )
            d.execute_callbacks()
            return True

        @ray.remote
        def start_sim(rank, chunk_pos, port):
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
            sys_md = {"world_size": 4, "master_address": "127.0.0.1", "master_port": port}
            try:
                b = Bridge(
                    bridge_id=rank,
                    arrays_metadata=arrays_md,
                    system_metadata=sys_md,
                    _node_id=None,
                    _comm_timeout=10,
                )  # type:ignore
                b.close(timestep=0)
            except Exception:
                raise

            return b.node_id, f"sched-{b.node_id}"

        # submit analytics first
        ref_analytics = head_script.remote()
        # submit sim after sleep_t seconds
        ref_sim = []

        # NOTE: only 3 nodes (so only 3 Bridges will be created)
        for i, w_nid in enumerate(worker_nodes[:3]):
            ref_sim.append(
                start_sim.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=w_nid, soft=False)).remote(
                    i, (i // 2, i % 2), port
                )
            )

        sim_res = ray.get(ref_sim)
        for i, (n_id, _) in enumerate(sim_res):
            assert n_id == worker_nodes[i]
        assert ray.get(ref_analytics)


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
