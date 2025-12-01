import dask.array as da
import ray

from tests.utils import ray_cluster, simple_worker, wait_for_head_node  # noqa: F401

NB_ITERATIONS = 100

@ray.remote(max_retries=0)
def head_script() -> None:
    """The head node checks that the values are correct"""
    from deisa.ray.window_api import Deisa
    from deisa.ray.types import WindowArrayDefinition

    deisa = Deisa()

    def simulation_callback(array: da.Array, *, timestep: int, preparation_result: da.Array):
        # We still have a dask array
        assert isinstance(preparation_result, da.Array)
        assert len(preparation_result.dask) == 1

        x_final = preparation_result.compute()
        assert x_final == 10 * timestep

    def prepare_iteration(array: da.Array, *, timestep: int) -> da.Array:
        # We can't use compute here since the data is not available yet
        return array.sum().persist()

    deisa.register_callback(
        simulation_callback,
        [WindowArrayDefinition("array")],
        max_iterations=NB_ITERATIONS,
        prepare_iteration=prepare_iteration,
        preparation_advance=10,
    )
    deisa.execute_callbacks()


def test_prepare_iteration(ray_cluster) -> None:  # noqa: F811
    head_ref = head_script.remote()
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
