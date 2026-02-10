import pytest
import ray

from deisa.ray.types import DeisaArray
from tests.utils import simple_worker_error_test, wait_for_head_node  # noqa: F401

NB_ITERATIONS = 10


@ray.remote(max_retries=0)
def head_script(enable_distributed_scheduling) -> None:
    """The head node checks that the values are correct"""
    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import WindowSpec
    import deisa.ray as deisa

    deisa.config.enable_experimental_distributed_scheduling(enable_distributed_scheduling)

    d = Deisa()

    # TODO : modify assert to test the actual error handler
    def simulation_callback(array: list[DeisaArray]):
        array[0].dask.compute()
        assert False

    d.register_callback(
        simulation_callback,
        [WindowSpec("array")],
    )
    d.execute_callbacks()


# WARNING : CRITICAL : Be careful if this test is failing, every other tests
#           can secretly be wrong without failing
@pytest.mark.parametrize(
    "nb_nodes, enable_distributed_scheduling",
    [
        (1, True),
        (1, False),
    ],
)
def test_exception_handler_not_bypass_computation(
    nb_nodes: int, enable_distributed_scheduling: bool, ray_cluster
) -> None:  # noqa: F811
    with pytest.raises(AssertionError):
        head_ref = head_script.remote(enable_distributed_scheduling)
        wait_for_head_node()

        worker_refs = []
        for rank in range(4):
            worker_refs.append(
                simple_worker_error_test.remote(
                    rank=rank,
                    position=(rank // 2, rank % 2),
                    chunks_per_dim=(2, 2),
                    nb_chunks_of_node=4 // nb_nodes,
                    chunk_size=(1, 1),
                    nb_iterations=NB_ITERATIONS,
                    node_id=f"node_{rank % nb_nodes}",
                    nb_nodes = 4,
                )
            )

        ray.get([head_ref] + worker_refs)


# @pytest.mark.parametrize(
#    "nb_nodes, enable_distributed_scheduling",
#    [
#        (1, True),
#    ],
# )
# def test_timeout_error(nb_nodes: int, enable_distributed_scheduling: bool, ray_cluster) -> None:  # noqa: F811
#    with pytest.raises(TimeoutError):
#        head_ref = head_script.remote(enable_distributed_scheduling)
#        wait_for_head_node()
#
#        worker_refs = []
#        for rank in range(4):
#            worker_refs.append(
#                simple_worker_error_test.remote(
#                    rank=rank,
#                    position=(rank // 2, rank % 2),
#                    chunks_per_dim=(2, 2),
#                    nb_chunks_of_node=4 // nb_nodes,
#                    chunk_size=(1, 1),
#                    nb_iterations=NB_ITERATIONS,
#                    node_id=f"node_{rank % nb_nodes}",
#                )
#            )
#
#        ray.get([head_ref] + worker_refs)
