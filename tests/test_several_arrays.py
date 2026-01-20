import dask.array as da
import ray
import pytest

from tests.utils import ray_cluster, simple_worker, wait_for_head_node  # noqa: F401

NB_ITERATIONS = 10


@ray.remote(max_retries=0)
def head_script(enable_distributed_scheduling) -> None:
    """The head node checks that the values are correct"""
    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import WindowSpec

    import deisa.ray as deisa

    deisa.config.enable_experimental_distributed_scheduling(enable_distributed_scheduling)

    d = Deisa()

    def simulation_callback(a: list[da.Array], b: list[da.Array]):
        timestep = a.t
        assert b[0].dask.sum().compute() == 20 * timestep

        assert len(b) == 1
        if timestep == 0:
            assert len(a) == 1
            return
        assert len(a) == 2

        assert a[0].dask.sum().compute() == 10 * (timestep - 1)
        assert a[1].dask.sum().compute() == 10 * timestep

        # Test a computation where the two arrays are used at the same time.
        # This checks that they are defined with different names.
        assert (a[1].dask - a[0].dask).sum().compute() == 10

    d.register_callback(
        simulation_callback,
        [
            WindowSpec("a", window_size=2),
            WindowSpec("b", window_size=1, preprocess=lambda x: 2 * x),
        ],
    )
    d.execute_callbacks()


@pytest.mark.parametrize("enable_distributed_scheduling", [False, True])
def test_several_arrays(enable_distributed_scheduling, ray_cluster) -> None:  # noqa: F811
    head_ref = head_script.remote(enable_distributed_scheduling)
    wait_for_head_node()

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
                array_name="a",
            )
        )

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
                array_name="b",
            )
        )

    ray.get([head_ref] + worker_refs)
