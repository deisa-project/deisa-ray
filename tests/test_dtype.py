import dask.array as da
import numpy as np
import ray
import pytest

from tests.utils import ray_cluster, simple_worker, wait_for_head_node  # noqa: F401


@ray.remote(max_retries=0)
def head_script(enable_distributed_scheduling) -> None:
    """The head node checks that the values are correct"""
    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import WindowArrayDefinition

    import deisa.ray as deisa

    deisa.config.enable_experimental_distributed_scheduling(enable_distributed_scheduling)

    d = Deisa()

    def simulation_callback(array: da.Array, timestep: int):
        assert array.dtype == np.int8

    d.register_callback(
        simulation_callback,
        [WindowArrayDefinition("array")],
        max_iterations=1,
    )
    d.execute_callbacks()


@pytest.mark.parametrize("enable_distributed_scheduling", [True, False])
def test_dtype(enable_distributed_scheduling, ray_cluster) -> None:  # noqa: F811
    head_ref = head_script.remote(enable_distributed_scheduling)
    wait_for_head_node()

    worker_ref = simple_worker.remote(
        rank=0,
        position=(0,),
        chunks_per_dim=(1,),
        nb_chunks_of_node=1,
        chunk_size=(1,),
        nb_iterations=1,
        node_id="node",
        dtype=np.int8,
    )

    ray.get([head_ref, worker_ref])
