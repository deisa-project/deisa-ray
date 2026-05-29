import os
import numpy as np
import ray
import pytest

from deisa.ray.types import DeisaArray
from tests.utils import WorkerSpec


@ray.remote(max_retries=0)
def head_script(enable_distributed_scheduling) -> None:
    """The head node checks that the values are correct"""
    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import Window

    os.environ["DEISA_DISTRIBUTED_SCHEDULING"] = "1" if enable_distributed_scheduling else "0"

    d = Deisa()

    def simulation_callback(array: list[DeisaArray]):
        assert array[0].dtype == np.int8

    d.register_callback(
        simulation_callback,
        *[Window("array")],
    )
    d.execute_callbacks()


@pytest.mark.parametrize("enable_distributed_scheduling", [True, False])
def test_dtype(enable_distributed_scheduling, ray_workflow) -> None:
    ray_workflow.start_head(head_script, enable_distributed_scheduling)
    ray_workflow.start_simple_workers(
        [
            WorkerSpec(
                rank=0,
                position=(0,),
                chunks_per_dim=(1,),
                chunk_size=(1,),
                nb_iterations=1,
                node_id="node",
                dtype=np.int8,
                nb_nodes=1,
            )
        ]
    )

    ray_workflow.wait()
