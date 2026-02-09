import pytest
import ray

from deisa.ray.types import DeisaArray
from tests.utils import ray_cluster, simple_worker, wait_for_head_node  # noqa: F401

NB_ITERATIONS = 10


@ray.remote(max_retries=0)
def head_script(enable_distributed_scheduling) -> None:
    """The head node checks that the values are correct"""
    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import WindowSpec
    import deisa.ray as deisa

    deisa.config.enable_experimental_distributed_scheduling(enable_distributed_scheduling)

    d = Deisa(n_sim_nodes=16)

    def simulation_callback1(array: list[DeisaArray]):
        x = array[0].dask.sum().compute()
        assert x == 136 * array[0].t

    def simulation_callback2(array1: list[DeisaArray]):
        x = array1[0].dask.sum().compute()
        assert x == 136 * array1[0].t

    def simulation_callback3(array: list[DeisaArray], array1: list[DeisaArray]):
        x = array[0].dask.sum().compute()
        y = array1[0].dask.sum().compute()
        assert x == 136 * array[0].t and y == 136 * array1[0].t

    d.register_callback(
        simulation_callback1,
        [WindowSpec("array")],
    )
    d.register_callback(
        simulation_callback2,
        [WindowSpec("array1")],
    )
    d.register_callback(
        simulation_callback3,
        [WindowSpec("array"), WindowSpec("array1")],
    )
    d.execute_callbacks()


@pytest.mark.parametrize(
    "enable_distributed_scheduling",
    [True, False],
)
def test_multiple_callbacks(enable_distributed_scheduling: bool, ray_cluster) -> None:  # noqa: F811
    head_ref = head_script.remote(enable_distributed_scheduling)
    wait_for_head_node()

    dim = 4
    worker_refs = []
    for rank in range(dim * dim):
        worker_refs.append(
            simple_worker.remote(
                rank=rank,
                position=(rank // dim, rank % dim),
                chunks_per_dim=(dim, dim),
                nb_chunks_of_node=1,
                chunk_size=(1, 1),
                nb_iterations=NB_ITERATIONS,
                node_id=f"node_{rank}",
                array_name=["array", "array1"],
            )
        )

    ray.get([head_ref] + worker_refs)


@ray.remote(max_retries=0)
def head_script2(enable_distributed_scheduling) -> None:
    """The head node checks that the values are correct"""
    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import WindowSpec
    import deisa.ray as deisa

    deisa.config.enable_experimental_distributed_scheduling(enable_distributed_scheduling)

    d = Deisa(n_sim_nodes=16)

    @d.callback(WindowSpec("array"))
    def simulation_callback1(array: list[DeisaArray]):
        x = array[0].dask.sum().compute()
        assert x == 136 * array[0].t

    @d.callback(WindowSpec("array1"))
    def simulation_callback2(array1: list[DeisaArray]):
        x = array1[0].dask.sum().compute()
        assert x == 136 * array1[0].t

    @d.callback(WindowSpec("array"), WindowSpec("array1"))
    def simulation_callback3(array: list[DeisaArray], array1: list[DeisaArray]):
        x = array[0].dask.sum().compute()
        y = array1[0].dask.sum().compute()
        assert x == 136 * array[0].t and y == 136 * array1[0].t

    d.execute_callbacks()


@pytest.mark.parametrize(
    "enable_distributed_scheduling",
    [True, False],
)
def test_multiple_callbacks_decorator(enable_distributed_scheduling: bool, ray_cluster) -> None:  # noqa: F811
    head_ref = head_script.remote(enable_distributed_scheduling)
    wait_for_head_node()

    dim = 4
    worker_refs = []
    for rank in range(dim * dim):
        worker_refs.append(
            simple_worker.remote(
                rank=rank,
                position=(rank // dim, rank % dim),
                chunks_per_dim=(dim, dim),
                nb_chunks_of_node=1,
                chunk_size=(1, 1),
                nb_iterations=NB_ITERATIONS,
                node_id=f"node_{rank}",
                array_name=["array", "array1"],
            )
        )

    ray.get([head_ref] + worker_refs)
