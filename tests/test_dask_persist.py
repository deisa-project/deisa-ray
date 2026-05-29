import os
import dask.array as da
import ray
from deisa.ray.types import DeisaArray
from tests.utils import WorkerSpec
import pytest


NB_ITERATIONS = 5


@ray.remote(max_retries=0)
def head_script(enable_distributed_scheduling) -> None:
    """The head node checks that the values are correct"""
    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import Window

    os.environ["DEISA_DISTRIBUTED_SCHEDULING"] = "1" if enable_distributed_scheduling else "0"

    d = Deisa()

    def simulation_callback(array: list[DeisaArray]):
        # This is the standard dask task graph
        assert len(array[0].sum().dask) == 9

        x = array[0].sum().persist()

        # We still have a dask array
        assert isinstance(x, da.Array)

        # But only one task in the task graph, since the result is being computed
        assert len(x.dask) == 1

        x_final = x.compute()
        assert x_final == 10 * array[0].t

    d.register_callback(
        simulation_callback,
        *[Window("array")],
    )
    d.execute_callbacks()


@pytest.mark.parametrize("enable_distributed_scheduling", [True, False])
def test_dask_persist(enable_distributed_scheduling, ray_workflow) -> None:
    nb_nodes = 4
    ray_workflow.start_head(head_script, enable_distributed_scheduling)
    ray_workflow.start_simple_workers(
        WorkerSpec(
            rank=rank,
            position=(rank // 2, rank % 2),
            chunks_per_dim=(2, 2),
            chunk_size=(1, 1),
            nb_iterations=NB_ITERATIONS,
            node_id=f"node_{rank}",
            nb_nodes=nb_nodes,
        )
        for rank in range(nb_nodes)
    )

    ray_workflow.wait()
