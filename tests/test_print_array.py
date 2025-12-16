import dask.array as da
import pytest
import ray
import numpy as np

from tests.utils import ray_cluster, simple_worker, wait_for_head_node  # noqa: F401

NB_ITERATIONS = 10


@ray.remote(max_retries=0)
def head_script(enable_distributed_scheduling) -> None:
    """The head node checks that the values are correct"""
    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import WindowArrayDefinition

    import deisa.ray as deisa

    deisa.config.enable_experimental_distributed_scheduling(enable_distributed_scheduling)

    d = Deisa()

    def simulation_callback(array: da.Array, timestep: int):
        x = array.compute()
        print(f"ARRAY PRINTED = {x}", flush=True)

        arr = timestep * np.array([[1,2],[3,4]])
        assert (arr == np.array(x)).all()

        # rebuilding with "wrong" structure
        # assert np.array2string(x) == np.array2string(arr)

        # NOTE: Test for 2d/3d/4d/... arrays

    d.register_callback(
        simulation_callback,
        [WindowArrayDefinition("array")],
        max_iterations=NB_ITERATIONS,
    )
    d.execute_callbacks()


# TODO: fix for distributed scheduling
@pytest.mark.parametrize(
        "nb_nodes, enable_distributed_scheduling", 
        [
            # (1, True),
            (1, False)
         ]
    )
def test_deisa_ray(nb_nodes: int, enable_distributed_scheduling, ray_cluster) -> None:  # noqa: F811
    head_ref = head_script.remote(enable_distributed_scheduling)
    wait_for_head_node()

    worker_refs = []
    for rank in range(4):
        worker_refs.append(
            simple_worker.remote(
                rank=rank,
                position=(rank // 2, rank % 2),
                chunks_per_dim=(2, 2),
                nb_chunks_of_node=4 // nb_nodes,
                chunk_size=(1, 1),
                nb_iterations=NB_ITERATIONS,
                node_id=f"node_{rank % nb_nodes}",
            )
        )

    ray.get([head_ref] + worker_refs)
