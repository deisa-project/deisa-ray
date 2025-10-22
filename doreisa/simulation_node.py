import logging
from typing import Callable

import numpy as np
import ray
import ray.actor

from doreisa.head_node import SimulationHead, get_head_actor_options


class Client:
    """
    Used by the MPI nodes to send data to the analytic cluster.

    The client is in charge of a several chunks of data. Each chunk is at a certain position in an
    array.
    """

    def __init__(self, *, _fake_node_id: str | None = None) -> None:
        """
        Args:
            _fake_node_id: The ID of the node. If None, the ID is taken from the Ray runtime context.
                This is useful for testing with several scheduling actors on a single machine.
        """
        if not ray.is_initialized():
            ray.init(address="auto", log_to_driver=False, logging_level=logging.ERROR)

        self.node_id = _fake_node_id or ray.get_runtime_context().get_node_id()

        # self.head = ray.get_actor("simulation_head", namespace="doreisa")
        
        # get head/simulation  actor, create it if not yet done by client 
        self.head  = SimulationHead.options(**get_head_actor_options()).remote()

        self.scheduling_actor: ray.actor.ActorHandle = ray.get(
            self.head.scheduling_actor.remote(self.node_id, is_fake_id=bool(_fake_node_id))
        )

        # Wait client  properly parametrized the head  actor
        ray.get(self.head.wait_init.remote())



        self.preprocessing_callbacks: dict[str, Callable] = ray.get(self.head.preprocessing_callbacks.remote())

    def add_chunk(
        self,
        array_name: str,
        chunk_position: tuple[int, ...],
        nb_chunks_per_dim: tuple[int, ...],
        nb_chunks_of_node: int,
        timestep: int,
        chunk: np.ndarray,
        store_externally: bool = False,
    ) -> None:
        """
        Make a chunk of data available to the analytic.

        Args:
            array_name: The name of the array.
            chunk_position: The position of the chunk in the array.
            nb_chunks_per_dim: The number of chunks per dimension.
            nb_chunks_of_node: The number of chunks sent by this node. The scheduling actor will
                inform the head actor when all the chunks are ready.
            chunk: The chunk of data.
            store_externally: If True, the data is stored externally. TODO Not implemented yet.
        """
        chunk = self.preprocessing_callbacks[array_name](chunk)

        # Setting the owner allows keeping the reference when the simulation script terminates.
        ref = ray.put(chunk, _owner=self.scheduling_actor)

        future: ray.ObjectRef = self.scheduling_actor.add_chunk.options(enable_task_events=False).remote(
            array_name,
            timestep,
            chunk_position,
            chunk.dtype,
            nb_chunks_per_dim,
            nb_chunks_of_node,
            [ref],
            chunk.shape,
        )  # type: ignore

        # Wait until the data is processed before returning to the simulation
        ray.get(future)
