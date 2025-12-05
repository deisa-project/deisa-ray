import pickle
import ray
from deisa.ray._async_dict import AsyncDict
from deisa.ray.types import ChunkRef, ScheduledByOtherActor, GraphInfo, ArrayPerTimestep, PartialArray
from deisa.ray.utils import get_ready_actor_with_retry
from deisa.ray.ray_patch import remote_ray_dask_get
from deisa.ray.errors import ContractError
from typing import Dict, Hashable, Any

class NodeBase:
    """
    Actor responsible for gathering chunks and exchanging data with analytics.

    Each node actor is associated with a specific node and is responsible for:

    - Collecting chunks of arrays sent by simulation nodes (via :class:`Bridge`)
    - Registering its owned chunks with the head node
    - Providing a small key/value channel (``set``, ``get``, ``delete``) for
      non-chunked feedback between analytics and simulation.

    The :class:`SchedulingActor` subclass adds graph-scheduling behaviour on
    top of this base functionality.

    Parameters
    ----------
    actor_id : int
        Unique identifier for this node actor, typically derived from the node
        ID.

    Attributes
    ----------
    actor_id : int
        The unique identifier for this node actor.
    actor_handle : ray.actor.ActorHandle
        Handle to this actor instance.
    head : ray.actor.ActorHandle
        Handle to the :class:`SimulationHead` actor.
    arrays : AsyncDict[str, Array]
        Dictionary mapping array names to their :class:`Array` instances,
        which track chunks and timesteps.
    feedback_non_chunked : dict
        Dictionary for storing non-chunked feedback values shared between
        analytics and simulation.
    """

    async def __init__(self, actor_id: int, arrays_metadata: Dict[str, Dict] = {}) -> None:
        self.actor_id = actor_id
        self.actor_handle = ray.get_runtime_context().current_actor

        self.head = get_ready_actor_with_retry(name="simulation_head", namespace="deisa_ray")
        await self.head.register_scheduling_actor.remote(actor_id, self.actor_handle)

        # Keeps track of array metadata AND ref per timestep. 
        # TODO: I think these two responsabilities could be separated.
        self.partial_arrays: AsyncDict[str, PartialArray] = AsyncDict()
        
        # For non-chunked feedback between analytics and simulation
        self.feedback_non_chunked: Dict[Hashable, Any] = {}

    def set(self,
            *args,
            key: Hashable,
            value: Any,
            chunked: bool = False,
            **kwargs
            )->None:
        if not chunked:
            self.feedback_non_chunked[key] = value
        else:
            # TODO: implement chunked version
            raise NotImplementedError()

    def get(self,
            key, 
            default = None,
            chunked = False,
            )->Any:
        if not chunked:
            return self.feedback_non_chunked.get(key, default)
        else:
            raise NotImplementedError()
        
    def delete(
        self,
        *args,
        key: Hashable,
        **kwargs,
    )->None:
        self.feedback_non_chunked.pop(key, None)

    def _create_or_retrieve_partial_array(self,
                                          array_name: str, 
                                          nb_chunks_of_node: int
                                          )->PartialArray:
        if array_name not in self.partial_arrays:
            self.partial_arrays[array_name] = PartialArray()
            self.nb_chunks_of_node = nb_chunks_of_node
        return self.partial_arrays[array_name]
        
    async def register_chunk(self, 
                             bridge_id: int, 
                             array_name: str, 
                             chunk_shape, 
                             nb_chunks_per_dim, 
                             nb_chunks_of_node: int, 
                             dtype, 
                             chunk_position
                             )->None:
        partial_array = self._create_or_retrieve_partial_array(array_name, nb_chunks_of_node)

        # add metadata for this array 
        partial_array.chunks_contained_meta.add((bridge_id, chunk_position, chunk_shape))

        # Technically, no race conditions should happen since its calls to the same actor method 
        # happen synchrnously (in ray). Since the method is async, it will run the method one a at 
        # a time, and give control to async runtime when it encounters await below. 
        # HOWEVER - with certain implementation of MPI, there have been problems here. 
        if len(partial_array.chunks_contained_meta) == nb_chunks_of_node:
            await self.head.register_partial_array.options(enable_task_events=False).remote(
                    self.actor_id,
                    array_name,
                    dtype,
                    # TODO I could figure this out from the global size and the chunk shape
                    nb_chunks_per_dim,
                    list(partial_array.chunks_contained_meta),
                )
            # per array async event
            partial_array.ready_event.set()
            partial_array.ready_event.clear()
        else:
            await partial_array.ready_event.wait()

    def preprocessing_callbacks(self) -> ray.ObjectRef:
        """
        Get the preprocessing callbacks for all arrays.

        Returns
        -------
        ray.ObjectRef
            ObjectRef to a dictionary mapping array names to their
            preprocessing callback functions.

        Notes
        -----
        This method returns an ObjectRef rather than the actual dictionary
        to avoid blocking. The callbacks are retrieved from the head node
        and used by Bridge instances to preprocess chunks before sending
        them to this node actor.
        """
        # return obect ref
        p_clbs = self.head.preprocessing_callbacks.remote()
        assert isinstance(p_clbs, ray.ObjectRef)
        return p_clbs

    def ready(self) -> None:
        """
        Check if the node actor is ready.

        Returns
        -------
        None
            Always returns None. This method serves as a readiness check
            for the actor.

        Notes
        -----
        This method can be called to verify that the actor has been
        successfully initialized and is ready to receive requests. It is
        used by `get_ready_actor_with_retry` to ensure the actor is
        operational before returning its handle.
        """
        pass

    def _pack_object_ref(self, refs: list[ray.ObjectRef]):
        """
        Pack a list of ObjectRefs into a single ObjectRef.

        This method is used to create an ObjectRef containing the given
        ObjectRefs, allowing the expected format in the task graph. It
        returns the first ObjectRef from the list.

        Parameters
        ----------
        refs : list[ray.ObjectRef]
            List of Ray object references to pack. Only the first one is
            returned.

        Returns
        -------
        ray.ObjectRef
            The first ObjectRef from the input list.

        Notes
        -----
        This is a method instead of a function with `num_cpus=0` to avoid
        starting many new workers. The method is called remotely to create
        the proper ObjectRef structure in the task graph.
        """
        return refs[0]

    # TODO: refactor from here
    async def add_chunk(
        self,
        bridge_id: int,
        array_name: str,
        chunk_ref: list[ray.ObjectRef],
        timestep: int,
        chunked: bool = True,
        *args,
        **kwargs
    ) -> None:
        """
        Add a chunk of data to this node actor.

        This method is called by Bridge instances to send chunks of arrays
        to this node actor. When all chunks from a node are received,
        the actor notifies the head node that chunks are ready.

        Parameters
        ----------
        # TODO fill up

        Notes
        -----
        This method manages chunk collection and coordination:
        1. Stores the chunk reference in the local chunks dictionary
        2. Records chunk ownership information
        3. When all chunks from the node are received:
           - Registers the array with the head node (first time only)
           - Collects all owned chunks and sends them to the head node
           - Converts chunk references to pickled bytes to free memory
           - Notifies the head node that chunks are ready
        4. Waits for the chunks_ready_event if chunks are not yet complete

        The method blocks until all chunks for this timestep from this node
        are ready, ensuring proper synchronization.
        """
        if array_name not in self.partial_arrays:
            # respect contract at the beginning
            raise ContractError(f"User requested to add chunk for {array_name} but this array has"  
                f"not been described. Please call register_array({array_name}) before calling"
                "add_chunk().")
        partial_array = self.partial_arrays[array_name]

        if timestep not in partial_array.per_timestep_arrays.keys():
            partial_array.per_timestep_arrays[timestep] = ArrayPerTimestep()
        array_timestep = partial_array.per_timestep_arrays[timestep]
        array_timestep.local_chunks[bridge_id] = self.actor_handle._pack_object_ref.remote(chunk_ref)

        if len(array_timestep.local_chunks) == self.nb_chunks_of_node:
            chunks = []
            for bridge_id, ref in array_timestep.local_chunks._data.items():
                assert isinstance(ref, ray.ObjectRef)
                chunks.append(ref)
                array_timestep.local_chunks[bridge_id] = pickle.dumps(ref)
            all_chunks_ref = ray.put(chunks)

            await self.head.chunks_ready.options(enable_task_events=False).remote(
                array_name, timestep, [all_chunks_ref]
            )
            array_timestep.chunks_ready_event.set()
            array_timestep.chunks_ready_event.clear()
        else:
            await array_timestep.chunks_ready_event.wait()

@ray.remote
class NodeActor(NodeBase):
    """
    Actor responsible for gathering chunks and exchanging data with analytics.

    This is a Ray actor. Shared logic is implemented in NodeBase.
    """

    async def __init__(self, actor_id: int, arrays_metadata: Dict[str, Dict] = {}) -> None:
        # Initialise the shared base part
        await NodeBase.__init__(self, actor_id=actor_id, arrays_metadata=arrays_metadata)
        # Optionally: NodeActor-specific init here


@ray.remote
class SchedulingActor(NodeBase):
    """
    Node actor with additional Dask graph scheduling behaviour.

    This actor inherits all chunk-collection and feedback mechanisms from
    :class:`NodeActor` and adds graph scheduling capabilities used by the
    custom Dask-on-Ray scheduler. When using a :class:`SchedulingActor`, the
    custom scheduler distributes Dask task graphs across multiple actors.
    When using a plain :class:`NodeActor`, standard Dask scheduling is used.

    Parameters
    ----------
    actor_id : int
        Unique identifier for this scheduling actor, typically the node ID.
    arrays_metadata : dict[str, dict], optional
        Currently unused but reserved for future extensions where the actor
        may need array-level metadata at construction time.

    Attributes
    ----------
    scheduling_actors : list[ray.actor.ActorHandle]
        List of all scheduling actor handles, populated lazily when first
        graph is scheduled.
    graph_infos : AsyncDict[int, GraphInfo]
        Dictionary mapping graph IDs to their :class:`GraphInfo` objects,
        which track graph scheduling state and results.
    """

    async def __init__(self, actor_id: int, arrays_metadata: Dict[str, Dict] = {}) -> None:
        # Delegate initialization to NodeActor, which sets up head node
        # registration, arrays, and feedback mechanisms.
        await super().__init__(actor_id=actor_id, arrays_metadata=arrays_metadata)
        
        # Scheduling-specific state (not needed for plain NodeActor)
        self.scheduling_actors: list[ray.actor.ActorHandle] = []
        self.graph_infos: AsyncDict[int, GraphInfo] = AsyncDict()

    async def schedule_graph(self, graph_id: int, dsk: dict) -> None:
        """
        Schedule a Dask task graph for execution.

        This method processes a Dask task graph, replacing placeholders
        (ChunkRef and ScheduledByOtherActor) with actual ObjectRefs, and
        schedules the graph for execution using Ray's Dask scheduler.

        Parameters
        ----------
        graph_id : int
            Unique identifier for this graph. Used to track the graph state
            and retrieve results later.
        dsk : dict
            The Dask task graph dictionary. Keys are task identifiers, and
            values can be:
            - Regular Dask tasks
            - ChunkRef objects (replaced with actual chunk ObjectRefs)
            - ScheduledByOtherActor objects (replaced with remote calls to
              other actors)

        Notes
        -----
        This method performs the following operations:
        1. Retrieves the list of all scheduling actors (if not already cached)
        2. Creates a GraphInfo object to track this graph's state
        3. Processes the task graph:
           - Replaces ScheduledByOtherActor with remote calls to other actors
           - Replaces ChunkRef with actual chunk ObjectRefs from local storage
           - Converts pickled chunk references back to ObjectRefs
        4. Schedules the graph using remote_ray_dask_get
        5. Stores the resulting ObjectRefs in GraphInfo
        6. Sets the scheduled_event to signal completion

        The method handles cross-actor dependencies by delegating tasks to
        the appropriate scheduling actors. Chunk references are retrieved
        asynchronously from the local chunks storage.
        """
        # Find the scheduling actors (lazy initialization)
        if not self.scheduling_actors:
            self.scheduling_actors = await self.head.list_scheduling_actors.options(enable_task_events=False).remote()

        # Create and store graph info for tracking this graph's execution
        info = GraphInfo()
        self.graph_infos[graph_id] = info

        for key, val in dsk.items():
            # Adapt external keys
            if isinstance(val, ScheduledByOtherActor):
                actor = self.scheduling_actors[val.actor_id]
                dsk[key] = actor.get_value.options(enable_task_events=False).remote(graph_id, key)

            # Replace the false chunks by the real ObjectRefs
            if isinstance(val, ChunkRef):
                assert val.actor_id == self.actor_id

                # TODO: maybe awaiting is not necessary. When would the key not be present in the 
                # AsyncDict?
                array = await self.partial_arrays.wait_for_key(val.array_name)

                array_timestep = await array.per_timestep_arrays.wait_for_key(val.timestep)

                ref = await array_timestep.local_chunks.wait_for_key(val.bridge_id)

                if isinstance(ref, bytes):  # This may not be the case depending on the asyncio scheduling order
                    ref = pickle.loads(ref)
                else:
                    ref = pickle.loads(pickle.dumps(ref))  # To free the memory automatically

                dsk[key] = ref

        # We will need the ObjectRefs of these keys
        keys_needed = list(dsk.keys())

        refs = await remote_ray_dask_get.remote(dsk, keys_needed)

        for key, ref in zip(keys_needed, refs):
            info.refs[key] = ref

        info.scheduled_event.set()

    async def get_value(self, graph_id: int, key: str):
        """
        Get the result value for a specific key from a scheduled graph.

        This method retrieves the Ray ObjectRef for a task key from a
        previously scheduled graph. It waits for the graph to be scheduled
        before returning the reference.

        Parameters
        ----------
        graph_id : int
            The identifier of the graph containing the key.
        key : str
            The task key to retrieve from the graph.

        Returns
        -------
        ray.ObjectRef
            Ray object reference to the result of the task with the given key.

        Notes
        -----
        This method is called by other scheduling actors when they need to
        retrieve values from graphs scheduled by this actor. It waits for
        the graph to be fully scheduled (via scheduled_event) before returning
        the ObjectRef. The method is used to handle cross-actor dependencies
        in distributed Dask computations.
        """
        graph_info = await self.graph_infos.wait_for_key(graph_id)

        await graph_info.scheduled_event.wait()
        return await graph_info.refs[key]
