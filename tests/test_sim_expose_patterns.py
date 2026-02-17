# Design different tests to see if various possible expose patterns are safe
# We have two arrays "A" and "B".

# Test 1: "typical" - One Bridge shares A and B, one right after the other (aka. "in sync"). This
# is the expected typical case.
# Test 2: "rank_ahead" - A is distributed among two bridges, B0 and B1. B0 starts and goes ahead by as many iters as it
# can. B1 had a late start to sending and begins after X seconds. Expected behavior is that analytics will block sim from
# doing this after some iterations due to the semaphore. Everything should work.
# Test 3: "out_of_sync" - 4 Bridges: Bridge_0(A), Bridge_1(A), Bridge_2(B), Bridge_3(B). B_0(A) starts a in parallel with
# B_2(B), then B_2(A) and B_3(B) start sending in parallel. We expect that much like Test2, analytics will block sim.
# Test 4: "out_of_sync_one_at_a_time" - 4 Bridges: Bridge_0(A), Bridge_1(A), Bridge_2(B), Bridge_3(B). B_0(A) first, then B_1(A),
# then B_2(B), then B_3(B). We expect that analytics will block sim.

import pytest
import ray

from deisa.ray.types import DeisaArray
from tests.utils import ray_cluster, simple_worker, wait_for_head_node, pick_free_port  # noqa: F401

NB_ITERATIONS = 5


@pytest.mark.parametrize(
    "nb_nodes, enable_distributed_scheduling",
    [
        (1, True),
        (1, False),
    ],
)
def test_typical(nb_nodes: int, enable_distributed_scheduling: bool, ray_cluster) -> None:  # noqa: F811
    @ray.remote(max_retries=0)
    def head_script(enable_distributed_scheduling) -> None:
        """The head node checks that the values are correct"""
        from deisa.ray.window_handler import Deisa
        from deisa.ray.types import WindowSpec
        import deisa.ray as deisa

        deisa.config.enable_experimental_distributed_scheduling(enable_distributed_scheduling)
        d = Deisa()

        @d.callback(WindowSpec("A"), WindowSpec("B"))
        def simulation_callback(A: list[DeisaArray], B: list[DeisaArray]):
            x = A[0].dask.sum().compute()
            y = B[0].dask.sum().compute()
            assert x == A[0].t
            assert y == B[0].t

        d.execute_callbacks()

    head_ref = head_script.remote(enable_distributed_scheduling)
    wait_for_head_node()
    port = pick_free_port()

    worker_refs = []
    worker_refs.append(
        simple_worker.remote(
            rank=0,
            position=(0, 0),
            chunks_per_dim=(1, 1),
            nb_chunks_of_node=1,
            chunk_size=(1, 1),
            nb_iterations=NB_ITERATIONS,
            node_id="node_0",
            nb_nodes=nb_nodes,
            array_name=["A", "B"],
            port=port,
        )
    )

    ray.get([head_ref] + worker_refs)

    # Check that the right number of scheduling actors were created
    simulation_head = ray.get_actor("simulation_head", namespace="deisa_ray")
    assert len(ray.get(simulation_head.list_scheduling_actors.remote())) == nb_nodes


@pytest.mark.parametrize(
    "nb_nodes, enable_distributed_scheduling",
    [
        (2, True),
        (2, False),
    ],
)
def test_rank_ahead(nb_nodes: int, enable_distributed_scheduling: bool, ray_cluster) -> None:  # noqa: F811
    @ray.remote(max_retries=0)
    def head_script(enable_distributed_scheduling) -> None:
        """The head node checks that the values are correct"""
        from deisa.ray.window_handler import Deisa
        from deisa.ray.types import WindowSpec
        import deisa.ray as deisa

        deisa.config.enable_experimental_distributed_scheduling(enable_distributed_scheduling)
        d = Deisa()

        @d.callback(WindowSpec("A"))
        def simulation_callback(A: list[DeisaArray]):
            x = A[0].dask.sum().compute()
            assert x == 3 * A[0].t

        d.execute_callbacks()

    head_ref = head_script.remote(enable_distributed_scheduling)
    wait_for_head_node()
    port = pick_free_port()

    worker_refs = []
    for rank in range(2):
        worker_refs.append(
            simple_worker.remote(
                rank=rank,
                position=(rank // 2, rank % 2),
                chunks_per_dim=(1, 2),
                nb_chunks_of_node=2 // nb_nodes,
                chunk_size=(1, 1),
                nb_iterations=NB_ITERATIONS,
                node_id=f"node_{rank % nb_nodes}",
                nb_nodes=nb_nodes,
                port=port,
                array_name=["A"],
                _sleep_b4_send=5,
            )
        )

    ray.get([head_ref] + worker_refs)

    # Check that the right number of scheduling actors were created
    simulation_head = ray.get_actor("simulation_head", namespace="deisa_ray")
    assert len(ray.get(simulation_head.list_scheduling_actors.remote())) == nb_nodes


@pytest.mark.parametrize(
    "nb_nodes, enable_distributed_scheduling",
    [
        (2, False),
        (2, True),
    ],
)
def test_out_of_sync(nb_nodes: int, enable_distributed_scheduling: bool, ray_cluster) -> None:  # noqa: F811
    @ray.remote(max_retries=0)
    def head_script(enable_distributed_scheduling) -> None:
        """The head node checks that the values are correct"""
        from deisa.ray.window_handler import Deisa
        from deisa.ray.types import WindowSpec
        import deisa.ray as deisa

        deisa.config.enable_experimental_distributed_scheduling(enable_distributed_scheduling)
        d = Deisa()

        @d.callback(WindowSpec("A"), WindowSpec("B"))
        def simulation_callback(A: list[DeisaArray], B: list[DeisaArray]):
            x = A[0].dask.sum().compute()
            y = B[0].dask.sum().compute()
            assert x == 3 * A[0].t
            assert y == 3 * B[0].t

        d.execute_callbacks()

    head_ref = head_script.remote(enable_distributed_scheduling)
    wait_for_head_node()
    port = pick_free_port()

    worker_refs = []
    worker_refs.append(
        simple_worker.remote(
            rank=0,
            position=(0, 0),
            chunks_per_dim=(1, 2),
            nb_chunks_of_node=1,
            chunk_size=(1, 1),
            nb_iterations=NB_ITERATIONS,
            node_id="node_0",
            nb_nodes=nb_nodes,
            port=port,
            array_name=["A", "B"],
        )
    )
    worker_refs.append(
        simple_worker.remote(
            rank=1,
            position=(0, 1),
            chunks_per_dim=(1, 2),
            nb_chunks_of_node=1,
            chunk_size=(1, 1),
            nb_iterations=NB_ITERATIONS,
            node_id="node_1",
            nb_nodes=nb_nodes,
            port=port,
            array_name=["A", "B"],
            _sleep_b4_send=5,
        )
    )

    ray.get([head_ref] + worker_refs)

    # Check that the right number of scheduling actors were created
    simulation_head = ray.get_actor("simulation_head", namespace="deisa_ray")
    assert len(ray.get(simulation_head.list_scheduling_actors.remote())) == nb_nodes


# NOTE : As Expected This test tests something else :
# RuntimeError: Logical flow of data was violated. Timestep 0 sent after timestep 4. Exiting...
#
# @pytest.mark.parametrize(
#     "nb_nodes, enable_distributed_scheduling",
#     [
#         (4, False),
#         (4, True),
#     ],
# )
# def test_out_of_sync_one_at_a_time(nb_nodes: int, enable_distributed_scheduling: bool, ray_cluster) -> None:  # noqa: F811
#     @ray.remote(max_retries=0)
#     def head_script(enable_distributed_scheduling) -> None:
#         """The head node checks that the values are correct"""
#         from deisa.ray.window_handler import Deisa
#         from deisa.ray.types import WindowSpec
#         import deisa.ray as deisa
#
#         deisa.config.enable_experimental_distributed_scheduling(enable_distributed_scheduling)
#         d = Deisa()
#
#         @d.callback(WindowSpec("A"), WindowSpec("B"))
#         def simulation_callback(A: list[DeisaArray], B: list[DeisaArray]):
#             x = A[0].dask.sum().compute()
#             y = B[0].dask.sum().compute()
#             assert x == 3 * A[0].t
#             assert y == 7 * B[0].t
#
#         d.execute_callbacks()
#
#     head_ref = head_script.remote(enable_distributed_scheduling)
#     wait_for_head_node()
#     port = pick_free_port()
#
#     worker_refs = [
#         simple_worker.remote(
#             rank=0,
#             position=(0, 0),
#             chunks_per_dim=(1, 2),
#             nb_chunks_of_node=1,
#             chunk_size=(1, 1),
#             nb_iterations=NB_ITERATIONS,
#             node_id="node_0",
#             nb_nodes=nb_nodes,
#             port=port,
#             array_name="A",
#         ),
#         simple_worker.remote(
#             rank=1,
#             position=(0, 1),
#             chunks_per_dim=(1, 2),
#             nb_chunks_of_node=1,
#             chunk_size=(1, 1),
#             nb_iterations=NB_ITERATIONS,
#             node_id="node_1",
#             nb_nodes=nb_nodes,
#             port=port,
#             array_name="A",
#             _sleep_b4_send=1,
#         ),
#     ]
#     worker_refs.append(
#         simple_worker.remote(
#             rank=2,
#             position=(0, 0),
#             chunks_per_dim=(1, 2),
#             nb_chunks_of_node=1,
#             chunk_size=(1, 1),
#             nb_iterations=NB_ITERATIONS,
#             node_id="node_2",
#             nb_nodes=nb_nodes,
#             port=port,
#             array_name="B",
#             _sleep_b4_send=10,
#         )
#     )
#     worker_refs.append(
#         simple_worker.remote(
#             rank=3,
#             position=(0, 1),
#             chunks_per_dim=(1, 2),
#             nb_chunks_of_node=1,
#             chunk_size=(1, 1),
#             nb_iterations=NB_ITERATIONS,
#             node_id="node_3",
#             nb_nodes=nb_nodes,
#             port=port,
#             array_name="B",
#             _sleep_b4_send=10,
#         )
#     )
#
#     ray.get([head_ref] + worker_refs)
#
#     # Check that the right number of scheduling actors were created
#     simulation_head = ray.get_actor("simulation_head", namespace="deisa_ray")
#     assert len(ray.get(simulation_head.list_scheduling_actors.remote())) == nb_nodes
