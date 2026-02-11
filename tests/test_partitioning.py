import dask.array as da
import pytest
import ray

from deisa.ray.types import DeisaArray
from tests.utils import ray_cluster, simple_worker, wait_for_head_node, pick_free_port  # noqa: F401


NB_ITERATIONS = 5


@ray.remote(max_retries=0)
def head_script(partitioning_strategy: str) -> None:
    """The head node checks that the values are correct"""
    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import WindowSpec

    import deisa.ray as deisa

    deisa.config.enable_experimental_distributed_scheduling(True)

    d = Deisa()

    def simulation_callback(array: list[DeisaArray]):
        x = array[0].dask.sum().compute(deisa_ray_partitioning_strategy=partitioning_strategy)
        assert x == 10 * array[0].t

        # Test with a full Dask computation
        assert da.ones((2, 2), chunks=(1, 1)).sum().compute(deisa_ray_partitioning_strategy=partitioning_strategy) == 4

    d.register_callback(
        simulation_callback,
        [WindowSpec("array")],
    )
    d.execute_callbacks()


@pytest.mark.parametrize("partitioning_strategy", ["random", "greedy"])
def test_partitioning(partitioning_strategy: str, ray_cluster) -> None:  # noqa: F811
    head_ref = head_script.remote(partitioning_strategy)
    wait_for_head_node()
    port = pick_free_port()

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
                node_id=f"node_{rank % 4}",
                nb_nodes=4,
                port=port,
            )
        )

    ray.get([head_ref] + worker_refs)
