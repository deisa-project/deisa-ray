import dask.array as da
import ray

from tests.utils import ray_cluster, simple_worker, wait_for_head_node  # noqa: F401

NB_ITERATIONS = 10


@ray.remote(max_retries=0)
def head() -> None:
    """The head node checks that the values are correct"""
    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import WindowArrayDefinition

    deisa = Deisa()

    def simulation_callback(a: list[da.Array], b: list[da.Array], timestep: int):
        assert b[0].sum().compute() == 20 * timestep

        assert len(b) == 1
        if timestep == 0:
            assert len(a) == 1
            return
        assert len(a) == 2

        assert a[0].sum().compute() == 10 * (timestep - 1)
        assert a[1].sum().compute() == 10 * timestep

        # Test a computation where the two arrays are used at the same time.
        # This checks that they are defined with different names.
        assert (a[1] - a[0]).sum().compute() == 10

    deisa.register_callback(
        simulation_callback,
        [
            WindowArrayDefinition("a", window_size=2),
            WindowArrayDefinition("b", window_size=1, preprocess=lambda x: 2 * x),
        ],
        max_iterations=NB_ITERATIONS,
    )
    deisa.execute_callbacks()


def test_several_arrays(ray_cluster) -> None:  # noqa: F811
    head_ref = head.remote()
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
