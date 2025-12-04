import logging
from typing import Callable, Type, Dict, Any

import numpy as np
import ray
import ray.actor
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from deisa.ray.scheduling_actor import SchedulingActor as _RealSchedulingActor
from deisa.ray.utils import get_ready_actor_with_retry


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
        print("FLAG0", flush=True)
        self.id = id
        self.arrays_metadata = arrays_metadata
        self.system_metadata = system_metadata
        print("FLAG1", flush=True)
        self.scheduling_actor: ray.actor.ActorHandle = ray.get_actor(
            "sched-central", namespace="deisa_ray")
        self.head: ray.actor.ActorHandle = get_ready_actor_with_retry(
            name="simulation_head", namespace="deisa_ray")
        print("FLAG2", flush=True)
        ray.get(self.head.ready.remote())
        print("FLAG3", flush=True)

        # check if ray.init has already been called.
        # Needed when starting ray cluster from python (mainly testing)
        if not ray.is_initialized():
            ray.init(address="auto", log_to_driver=False,
                     logging_level=logging.ERROR)

        self.node_id = _node_id or ray.get_runtime_context().get_node_id()

        for array_name, meta in arrays_metadata.items():
            self.scheduling_actor.register_chunk.remote(
                bridge_id=self.id,
                array_name=array_name,
                chunk_shape=meta["chunk_shape"],
                nb_chunks_per_dim=meta["nb_chunks_per_dim"],
                nb_chunks_of_node=meta["nb_chunks_of_node"],
                dtype=meta["dtype"],
                chunk_position=meta["chunk_position"],
            )

        # TODO: hide preprocessing_callbacks

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

    def send(
        self,
        *args,
        array_name: str,
        chunk: np.ndarray,
        timestep: int,
        chunked: bool = True,
        store_externally: bool = False,
        **kwargs
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
        if chunked:
            pass
        chunk = self.preprocessing_callbacks[array_name](chunk)

        # Setting the owner allows keeping the reference when the simulation script terminates.
        ref = ray.put(chunk, _owner=self.scheduling_actor)

        future: ray.ObjectRef = self.scheduling_actor.send.remote(
            bridge_id=self.id,
            array_name=array_name,
            chunk_ref=[ref],
            timestep=timestep,
            chunked=True,
            store_externally=False,
        )  # type: ignore
        print("FLAG4", flush=True)

        # Wait until the data is processed before returning to the simulation
        ray.get(future)

    def get(self, *args, name: str, default: Any = None, chunked: bool = False, **kwargs) -> Any | None:
        """
        Retrieve information back from Analytics. 

        Used for two cases: 
        1) Retrieve a simple value that is set in the Analytics so that the simulation can react 
        to some event that has been detected. This case is asynchronous.
        2) Retrieve the same distributed array that has been modified somehow by the Analytics. 
        This case is synchronous. 

        Parameters
        ----------
        name : str
            The name of the key that is being retrieved from the Analytics.
        default : Any
            The default value to return if the key has not been set or does not exist.
        chunked : bool
            Whether the value that is returned is distributed or not. Should be set to True, only if 
            retrieving a distributed array that is handled by the Bridge.

        Notes
        -----
        TODO: Fill notes
        """
        if not chunked:
            return ray.get(self.scheduling_actor.get.remote(name, default, chunked))
        else:
            raise NotImplementedError()

    def _delete(self, *args, name: str, **kwargs):
        """
        Delete a key from the information shared by Analytics. 

        Should be called immediately after something has been shared. This is done so that the user 
        is in charge of handling events and the simulation does not keep detecting the same 
        event.

        Parameters
        ----------
        name : str
            The name of the key that is being retrieved from the Analytics.

        Notes
        -----
        TODO: Should we make this private or allow user to handle it? For now, we decided that 
        we will call it ourselves after every get, so error should never be raised.

        Raises
        ------
        KeyError
            If `name` does not exist among the keys that have been shared by analytics.
        """
        pass
