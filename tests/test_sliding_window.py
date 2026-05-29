import os
import ray
import pytest

from deisa.ray.types import DeisaArray
from tests.utils import WorkerSpec

NB_ITERATIONS = 5


@ray.remote(max_retries=0)
def head_script(enable_distributed_scheduling) -> None:
    """The head node checks that the values are correct"""
    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import Window

    os.environ["DEISA_DISTRIBUTED_SCHEDULING"] = "1" if enable_distributed_scheduling else "0"

    d = Deisa()

    def simulation_callback(array: list[DeisaArray]):
        if array[-1].t == 0:
            assert len(array) == 1
            return

        assert array[0].sum().compute() == 10 * array[0].t
        assert array[1].sum().compute() == 10 * array[1].t

        # Test a computation where the two arrays are used at the same time.
        # This checks that they are defined with different names.
        assert (array[1] - array[0]).sum().compute() == 10

    d.register_callback(
        simulation_callback,
        *[
            Window("array", size=2),
        ],
    )
    d.execute_callbacks()


@pytest.mark.parametrize("enable_distributed_scheduling", [False, True])
def test_sliding_window(enable_distributed_scheduling, ray_workflow) -> None:
    ray_workflow.start_head(head_script, enable_distributed_scheduling)
    ray_workflow.start_simple_workers(
        WorkerSpec(
            rank=rank,
            position=(rank // 2, rank % 2),
            chunks_per_dim=(2, 2),
            chunk_size=(1, 1),
            nb_iterations=NB_ITERATIONS,
            node_id=f"node_{rank}",
            nb_nodes=4,
        )
        for rank in range(4)
    )

    ray_workflow.wait()
