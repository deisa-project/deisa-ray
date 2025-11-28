import logging
from typing import Callable, Type, Dict

import numpy as np
import ray
import ray.actor
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from deisa.ray._scheduling_actor import SchedulingActor as _RealSchedulingActor



class Bridge:
    """
    Bridge between MPI ranks and Ray cluster for distributed array processing.

    Each Bridge instance is created by an MPI rank to connect to the Ray cluster
    and send data chunks. Each Bridge is responsible for managing a chunk of data
    from the decomposed distributed array.

    Parameters
    ----------
    _node_id : str or None, optional
        The ID of the node. If None, the ID is taken from the Ray runtime context.
        Useful for testing with several scheduling actors on a single machine.
        Default is None.
    scheduling_actor_cls : Type, optional
        The class to use for creating the scheduling actor. Default is
        `_RealSchedulingActor`.
    _init_retries : int, optional
        Number of retry attempts when initializing the scheduling actor.
        Default is 3.

    Attributes
    ----------
    node_id : str
        The ID of the node this Bridge is associated with.
    scheduling_actor : ray.actor.ActorHandle
        The Ray actor handle for the scheduling actor.
    preprocessing_callbacks : dict[str, Callable]
        Dictionary mapping array names to their preprocessing callback functions.

    Notes
    -----
    The Bridge automatically initializes Ray if it hasn't been initialized yet.
    The scheduling actor is created with a detached lifetime to persist beyond
    the Bridge initialization. The actor uses node affinity scheduling to ensure
    it runs on the specified node.

    Examples
    --------
    >>> bridge = Bridge()
    >>> bridge.add_chunk(
    ...     array_name="temperature",
    ...     chunk_position=(0, 0),
    ...     nb_chunks_per_dim=(2, 2),
    ...     nb_chunks_in_node=1,
    ...     timestep=0,
    ...     chunk=np.array([[1.0, 2.0], [3.0, 4.0]])
    ... )
    """

    def __init__(
        self,
        id: int,
        arrays_metadata: Dict[str, Dict],
        system_metadata: Dict,
        *args,
        _node_id: str | None = None,
        scheduling_actor_cls: Type = _RealSchedulingActor,
        _init_retries: int = 3,
        **kwargs
    ) -> None:
        """
        Initialize the Bridge to connect MPI rank to Ray cluster.

        Parameters
        ----------
        id : int
            Unique identifier of this Bridge. 
        arrays_metadata : Dict[str, Dict]
            Dictionary that describes the arrays being shared by the simulation. Keys represent the 
            name of the array while the values are dictionaries that must at least declare the 
            global size of that array.
        system_metadata : Dict 
            System metadata such as address of Ray cluster, number of MPI ranks, and other general 
            information that describes the system.
        _node_id : str or None, optional
            The ID of the node. If None, the ID is taken from the Ray runtime
            context. Useful for testing with several scheduling actors on a
            single machine. Default is None.
        scheduling_actor_cls : Type, optional
            The class to use for creating the scheduling actor. Default is
            `_RealSchedulingActor`.
        _init_retries : int, optional
            Number of retry attempts when initializing the scheduling actor.
            Default is 3.

        Raises
        ------
        RuntimeError
            If the scheduling actor cannot be created or initialized after
            the specified number of retries.

        Notes
        -----
        This method automatically initializes Ray if it hasn't been initialized
        yet. The scheduling actor is created with a detached lifetime and uses
        node affinity scheduling when `_node_id` is None. The first remote call
        to the scheduling actor serves as a readiness check.
        """
        # check if ray.init has already been called.
        # Needed when starting ray cluster from python (mainly testing)

        if not ray.is_initialized():
            ray.init(address="auto", log_to_driver=False, logging_level=logging.ERROR)

        self.node_id = _node_id or ray.get_runtime_context().get_node_id()
        name = f"sched-{self.node_id}"
        namespace = "deisa_ray"

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
                # name: "sched-{node_id}", namespace: "deisa_ray", node_id: {node_id}
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

        This method applies preprocessing callbacks to the chunk, stores it in
        Ray's object store, and sends it to the scheduling actor. The method
        blocks until the data is processed by the scheduling actor.

        Parameters
        ----------
        array_name : str
            The name of the array this chunk belongs to.
        chunk_position : tuple[int, ...]
            The position of the chunk in the array decomposition, specified as
            a tuple of indices for each dimension.
        nb_chunks_per_dim : tuple[int, ...]
            The number of chunks per dimension in the array decomposition.
        nb_chunks_in_node : int
            The number of chunks sent by this node. The scheduling actor will
            inform the head actor when all chunks from this node are ready.
        timestep : int
            The timestep index for this chunk of data.
        chunk : np.ndarray
            The chunk of data to be sent to the analytic.
        store_externally : bool, optional
            If True, the data is stored externally. Not implemented yet.
            Default is False.

        Notes
        -----
        The chunk is first processed through the preprocessing callback
        associated with `array_name`. The processed chunk is then stored in
        Ray's object store with the scheduling actor as the owner, ensuring
        the reference persists even after the simulation script terminates.
        This method blocks until the scheduling actor has processed the chunk.

        Raises
        ------
        KeyError
            If `array_name` is not found in the preprocessing callbacks
            dictionary.
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
