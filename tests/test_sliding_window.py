import ray
import pytest

from deisa.ray.types import DeisaArray
from tests.utils import ray_cluster, simple_worker, wait_for_head_node, pick_free_port  # noqa: F401

NB_ITERATIONS = 5


@ray.remote(max_retries=0)
def head_script(enable_distributed_scheduling) -> None:
    """The head node checks that the values are correct"""
    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import WindowSpec
    import deisa.ray as deisa

    deisa.config.enable_experimental_distributed_scheduling(enable_distributed_scheduling)

    d = Deisa()

    def simulation_callback(array: list[DeisaArray]):
        if array[-1].t == 0:
            assert len(array) == 1
            return

        assert array[0].dask.sum().compute() == 10 * array[0].t
        assert array[1].dask.sum().compute() == 10 * array[1].t

        # Test a computation where the two arrays are used at the same time.
        # This checks that they are defined with different names.
        assert (array[1].dask - array[0].dask).sum().compute() == 10

    d.register_callback(
        simulation_callback,
        [
            WindowSpec("array", window_size=2),
        ],
    )
    d.execute_callbacks()


@pytest.mark.parametrize("enable_distributed_scheduling", [False, True])
def test_sliding_window(enable_distributed_scheduling, ray_cluster) -> None:  # noqa: F811
    head_ref = head_script.remote(enable_distributed_scheduling)
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
                node_id=f"node_{rank}",
                nb_nodes=4,
                port=port,
            )
        )

    ray.get([head_ref] + worker_refs)
