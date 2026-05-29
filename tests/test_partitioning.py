import os
import dask.array as da
import pytest
import ray

from deisa.ray.types import DeisaArray
from tests.utils import WorkerSpec


NB_ITERATIONS = 5


@ray.remote(max_retries=0)
def head_script(partitioning_strategy: str) -> None:
    """The head node checks that the values are correct"""
    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import Window

    os.environ["DEISA_DISTRIBUTED_SCHEDULING"] = "1"

    d = Deisa()

    def simulation_callback(array: list[DeisaArray]):
        x = array[0].sum().compute(deisa_ray_partitioning_strategy=partitioning_strategy)
        assert x == 10 * array[0].t

        # Test with a full Dask computation
        assert da.ones((2, 2), chunks=(1, 1)).sum().compute(deisa_ray_partitioning_strategy=partitioning_strategy) == 4

    d.register_callback(
        simulation_callback,
        *[Window("array")],
    )
    d.execute_callbacks()


@pytest.mark.parametrize("partitioning_strategy", ["random", "greedy"])
def test_partitioning(partitioning_strategy: str, ray_workflow) -> None:
    ray_workflow.start_head(head_script, partitioning_strategy)
    ray_workflow.start_simple_workers(
        WorkerSpec(
            rank=rank,
            position=(rank // 2, rank % 2),
            chunks_per_dim=(2, 2),
            chunk_size=(1, 1),
            nb_iterations=NB_ITERATIONS,
            node_id=f"node_{rank % 4}",
            nb_nodes=4,
        )
        for rank in range(4)
    )

    ray_workflow.wait()
