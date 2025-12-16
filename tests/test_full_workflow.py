import dask.array as da
import pytest
import ray

from tests.utils import ray_cluster, simple_worker, wait_for_head_node  # noqa: F401

NB_ITERATIONS = 10


@ray.remote(max_retries=0)
def head_script() -> None:
    """The head node checks that the values are correct"""
    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import WindowArrayDefinition

    deisa = Deisa()

    def simulation_callback(array: da.Array, timestep: int):
        x = array.sum().compute()

        assert x == 10 * timestep

    deisa.register_callback(
        simulation_callback,
        [WindowArrayDefinition("array")],
        max_iterations=NB_ITERATIONS,
    )
    deisa.execute_callbacks()


@pytest.mark.parametrize("nb_nodes", [1, 2, 4])
def test_deisa_ray(nb_nodes: int, ray_cluster) -> None:  # noqa: F811
    head_ref = head_script.remote()
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

    # Check that the right number of scheduling actors were created
    simulation_head = ray.get_actor("simulation_head", namespace="deisa_ray")
    assert len(ray.get(simulation_head.list_scheduling_actors.remote())) == nb_nodes
