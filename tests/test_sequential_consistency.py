import pytest
import ray

from deisa.ray.types import DeisaArray
from tests.utils import wait_for_head_node  # noqa: F401
import numpy as np

NB_ITERATIONS = 10


@ray.remote(num_cpus=0, max_retries=0)
def strange_worker(
    *,
    rank: int,
    position: tuple[int, ...],
    chunks_per_dim: tuple[int, ...],
    nb_chunks_of_node: int,
    chunk_size: tuple[int, ...],
    nb_iterations: int,
    node_id: str | None = None,
    array_name: str | list[str] = "array",
    dtype: np.dtype = np.int32,  # type: ignore
    **kwargs,
) -> None:
    """Strange worker that sends nodes out of order!"""
    from deisa.ray.bridge import Bridge
    from deisa.ray.utils import get_system_metadata

    if isinstance(array_name, str):
        array_name = [array_name]

    start_iteration = kwargs.get("start_iteration", 0)

    sys_md = get_system_metadata()
    arrays_md = {
        name: {
            "chunk_shape": chunk_size,
            "nb_chunks_per_dim": chunks_per_dim,
            "nb_chunks_of_node": nb_chunks_of_node,
            "dtype": dtype,
            "chunk_position": position,
        }
        for name in array_name
    }

    client = Bridge(bridge_id=rank, arrays_metadata=arrays_md, system_metadata=sys_md, _node_id=node_id)

    array = (rank + 1) * np.ones(chunk_size, dtype=dtype)

    for i in range(start_iteration, nb_iterations // 2):
        for array_described in list(arrays_md.keys()):
            chunk = i * array
            client.send(array_name=array_described, chunk=chunk, timestep=i, chunked=True)
    mid_t = i
    # skip 2 iterations
    i += 3
    for array_described in list(arrays_md.keys()):
        chunk = i * array
        client.send(array_name=array_described, chunk=chunk, timestep=i, chunked=True)
    # send rest from mid_t (duplicating a step but doesnt matter)
    for i in range(mid_t, nb_iterations):
        for array_described in list(arrays_md.keys()):
            chunk = i * array
            client.send(array_name=array_described, chunk=chunk, timestep=i, chunked=True)
    client.close(timestep=nb_iterations)


@ray.remote(max_retries=0)
def head_script(enable_distributed_scheduling, nb_nodes) -> None:
    """The head node checks that the values are correct"""
    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import WindowSpec
    import deisa.ray as deisa

    deisa.config.enable_experimental_distributed_scheduling(enable_distributed_scheduling)

    d = Deisa(n_sim_nodes=nb_nodes)

    def simulation_callback(array: list[DeisaArray]):
        pass

    d.register_callback(
        simulation_callback,
        [WindowSpec("array")],
    )
    d.execute_callbacks()


@pytest.mark.parametrize(
    "nb_nodes, enable_distributed_scheduling",
    [
        (4, True),
        (4, False),
    ],
)
def test_arrays_sent_out_of_order_fails_analytics(
    nb_nodes: int, enable_distributed_scheduling: bool, ray_cluster
) -> None:  # noqa: F811
    with pytest.raises(RuntimeError):
        head_ref = head_script.remote(enable_distributed_scheduling, nb_nodes)
        wait_for_head_node()

        worker_refs = []
        for rank in range(4):
            worker_refs.append(
                strange_worker.remote(
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
