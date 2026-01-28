import dask.array as da
import ray
from deisa.ray.types import DeisaArray
from tests.utils import ray_cluster, simple_worker, wait_for_head_node  # noqa: F401
import pytest

NB_ITERATIONS = 10


@ray.remote(max_retries=0)
def head_script(enable_distributed_scheduling) -> None:
    """The head node checks that the values are correct"""
    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import WindowSpec

    import deisa.ray as deisa

    deisa.config.enable_experimental_distributed_scheduling(enable_distributed_scheduling)

    d = Deisa(n_sim_nodes=4)

    def simulation_callback(array: list[DeisaArray]):
        # This is the standard dask task graph
        assert len(array[0].dask.sum().dask) == 9

        x = array[0].dask.sum().persist()

        # We still have a dask array
        assert isinstance(x, da.Array)

        # But only one task in the task graph, since the result is being computed
        assert len(x.dask) == 1

        x_final = x.compute()
        assert x_final == 10 * array[0].t

    d.register_callback(
        simulation_callback,
        [WindowSpec("array")],
    )
    d.execute_callbacks()


@pytest.mark.parametrize("enable_distributed_scheduling", [True, False])
def test_dask_persist(enable_distributed_scheduling, ray_cluster) -> None:  # noqa: F811
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
            )
        )

    ray.get([head_ref] + worker_refs)
