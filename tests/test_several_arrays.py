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

    def simulation_callback(a: list[DeisaArray], b: list[DeisaArray]):
        assert b[0].sum().compute() == 10 * b[0].t

        assert len(b) == 1
        if a[-1].t == 0:
            assert len(a) == 1
            return
        assert len(a) == 2

        assert a[0].sum().compute() == 10 * a[0].t
        assert a[1].sum().compute() == 10 * a[1].t

        # Test a computation where the two arrays are used at the same time.
        # This checks that they are defined with different names.
        assert (a[1] - a[0]).sum().compute() == 10

    d.register_callback(
        simulation_callback,
        *[
            Window("a", size=2),
            Window("b", size=1),
        ],
    )
    d.execute_callbacks()


@pytest.mark.parametrize("enable_distributed_scheduling", [False, True])
def test_several_arrays(enable_distributed_scheduling, ray_workflow) -> None:
    ray_workflow.start_head(head_script, enable_distributed_scheduling)
    ray_workflow.start_simple_workers(
        WorkerSpec(
            rank=rank,
            position=(rank // 2, rank % 2),
            chunks_per_dim=(2, 2),
            chunk_size=(1, 1),
            nb_iterations=NB_ITERATIONS,
            node_id=f"node_{rank}",
            array_name=["a", "b"],
            nb_nodes=4,
        )
        for rank in range(4)
    )

    ray_workflow.wait()
