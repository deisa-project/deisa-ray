import os
import pytest
import ray
import numpy as np

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
        x = array[0].compute()
        print(f"ARRAY PRINTED = {x}", flush=True)

        arr = array[0].t * np.array([[1, 2], [3, 4]])
        assert (arr == np.array(x)).all()

        # TODO: Test for 2d/3d/4d/... arrays

    d.register_callback(
        simulation_callback,
        *[Window("array")],
    )
    d.execute_callbacks()


@pytest.mark.parametrize(
    "nb_nodes, enable_distributed_scheduling",
    [(1, True), (1, False)],
)
def test_deisa_ray(nb_nodes: int, enable_distributed_scheduling, ray_workflow) -> None:
    ray_workflow.start_head(head_script, enable_distributed_scheduling)
    ray_workflow.start_simple_workers(
        WorkerSpec(
            rank=rank,
            position=(rank // 2, rank % 2),
            chunks_per_dim=(2, 2),
            chunk_size=(1, 1),
            nb_iterations=NB_ITERATIONS,
            node_id=f"node_{rank % nb_nodes}",
            nb_nodes=4,
        )
        for rank in range(4)
    )

    ray_workflow.wait()
