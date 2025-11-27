import dask.array as da
import numpy as np
import ray

from tests.utils import ray_cluster, simple_worker, wait_for_head_node  # noqa: F401


@ray.remote(max_retries=0)
def head_script() -> None:
    """The head node checks that the values are correct"""
    from deisa.ray.head_node import init
    from deisa.ray.window_api import ArrayDefinition, run_simulation

    init()

    def simulation_callback(array: da.Array, timestep: int):
        assert array.dtype == np.int8

    run_simulation(
        simulation_callback,
        [ArrayDefinition("array")],
        max_iterations=1,
    )


def test_dtype(ray_cluster) -> None:  # noqa: F811
    head_ref = head_script.remote()
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
