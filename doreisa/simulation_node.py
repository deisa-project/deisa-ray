import logging
from typing import Callable, Type

import numpy as np
import ray
import ray.actor
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from doreisa._scheduling_actor import SchedulingActor as _RealSchedulingActor


class Client:
    """
    Instantiated by each MPI node to send data to the Ray cluster which will subsequently perform
    the requested analytics.

    The client is in charge of a several *local* chunks of data which are part of a global array. Each chunk has
    a specific position within the global array.
    """

    def __init__(
        self,
        *,
        _node_id: str | None = None,
        scheduling_actor_cls: Type = _RealSchedulingActor,
        _init_retries: int = 3,
    ) -> None:
        """
        Args:
            _node_id: The ID of the node. If None, the ID is taken from the Ray runtime context.
                Useful for testing with several scheduling actors on a single machine.
        """
        # check if ray.init has already been called.
        # Needed when starting ray cluster from python (mainly testing)

        if not ray.is_initialized():
            ray.init(address="auto", log_to_driver=False, logging_level=logging.ERROR)

        self.node_id = _node_id or ray.get_runtime_context().get_node_id()
        name = f"sched-{self.node_id}"
        namespace = "doreisa"

        # NOTE: now lifetime is detached, otherwise at the end of the init, it will get killed
        # NOTE: get_if_exists prevents race conditions
        scheduling_actor_options = {
            "name": name,
            "namespace": namespace,
            "lifetime": "detached",
            "get_if_exists": True,
            # WARNING: be careful - if not using async actor (has at least one async method)
            # this will make OS try to spawn 1 billion threads and it will blow up
            # StubSchedulingActor is async because of this.
            "max_concurrency": 1000_000_000,
            "num_cpus": 0,
            "enable_task_events": False,
        }

        if _node_id is None:
            scheduling_actor_options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
                node_id=self.node_id, soft=False
            )

        last_err = None
        for _ in range(max(1, _init_retries)):
            try:
                # first rank to arrive here will try to create scheduling actor. Ray will guarantee
                # that only one wil be created bc of get_if_exists. No need to use async events
                # creates actor with:
                # name: "sched-{node_id}", namespace: "doreisa", node_id: {node_id}
                self.scheduling_actor: ray.actor.ActorHandle = scheduling_actor_cls.options(
                    **scheduling_actor_options
                ).remote(actor_id=self.node_id)  # type: ignore

                # "Readiness" gate: first RPC must succeed. This means the scheduling_actor is
                # created and operational. No need to have a "ready" method.
                # NOTE: scheduling actor does head.preprocessing_callbacks.remote() which is a ref
                # we don't need the actual data there.
                # scheduling_actor.preprocessing_callbacks.remote() gives back another ref. The
                # first ray.get() is to get the result of the remote call. the second ray.get() is
                # to dereference the original ref.
                self.preprocessing_callbacks: dict[str, Callable] = ray.get(
                    ray.get(
                        self.scheduling_actor.preprocessing_callbacks.remote()  # type: ignore
                    )
                )

                # assert we have a dict for the preprocessing callbacks
                # TODO: preprocessing_callbacks are static for now. In the future it could be nice
                # to support ability to change them
                assert isinstance(self.preprocessing_callbacks, dict)

                break  # success
            except Exception as e:  # ray.exceptions.RayActorError and friends
                last_err = e

                # Try to re-create a fresh actor instance (same name will resolve to existing or new one)
                # Small backoff; no sleep needed for tests, but harmless if added.
                continue
        # `else:` clause belongs to for loop, and is only executed if it finishes normally without
        # a encountering a `break` statement (in our case it means the actor was never created).
        else:
            # keep original error
            raise RuntimeError(f"Failed to create/ready scheduling actor for node {self.node_id}") from last_err

    def add_chunk(
        self,
        array_name: str,
        chunk_position: tuple[int, ...],
        nb_chunks_per_dim: tuple[int, ...],
        nb_chunks_in_node: int,
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

        future: ray.ObjectRef = self.scheduling_actor.add_chunk.remote(
            array_name,
            timestep,
            chunk_position,
            chunk.dtype,
            nb_chunks_per_dim,
            nb_chunks_in_node,
            [ref],
            chunk.shape,
        )  # type: ignore

        # Wait until the data is processed before returning to the simulation
        ray.get(future)
