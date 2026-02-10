import pytest
import ray

from deisa.ray.types import DeisaArray
from tests.utils import ray_cluster, simple_worker, wait_for_head_node  # noqa: F401

NB_ITERATIONS = 10
START_ITERATION = 3


@ray.remote(max_retries=0)
def head_script(enable_distributed_scheduling) -> None:
    """The head node checks that the values are correct"""
    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import WindowSpec
    import deisa.ray as deisa

    deisa.config.enable_experimental_distributed_scheduling(enable_distributed_scheduling)

    d = Deisa()

    def simulation_callback(array: list[DeisaArray]):
        if len(array) == 1:
            assert array[0].t == START_ITERATION

        array[0].dask.compute()

    d.register_callback(
        simulation_callback,
        [WindowSpec("array", window_size=2)],
    )
    d.execute_callbacks()


@pytest.mark.parametrize(
    "enable_distributed_scheduling",
    [False, True],
)
def test_start_any_timestep(enable_distributed_scheduling: bool, ray_cluster) -> None:  # noqa: F811
    head_ref = head_script.remote(enable_distributed_scheduling)
    wait_for_head_node()

    worker_refs = []
    for rank in range(4):
        worker_refs.append(
            simple_worker.remote(
                rank=rank,
                position=(rank // 2, rank % 2),
                chunks_per_dim=(2, 2),
                nb_chunks_of_node=1,
                chunk_size=(1, 1),
                nb_iterations=NB_ITERATIONS,
                node_id=f"node_{rank}",
                start_iteration=START_ITERATION,
                nb_nodes = 4,
            )
        )

    ray.get([head_ref] + worker_refs)
