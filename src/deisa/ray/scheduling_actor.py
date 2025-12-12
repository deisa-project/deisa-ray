import pickle
import ray
from deisa.ray._async_dict import AsyncDict
from deisa.ray.types import (
    GraphKey,
    GraphValue,
    ChunkRef,
    ScheduledByOtherActor,
    GraphInfo,
    ArrayPerTimestep,
    PartialArray,
    DoubleRef,
    RayActorHandle,
    ActorID,
)
from deisa.ray.utils import get_ready_actor_with_retry
from deisa.ray.ray_patch import remote_ray_dask_get
from deisa.ray.errors import ContractError
from typing import Dict, Hashable, Any


class NodeActorBase:
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
    actor_handle : RayActorHandle
        Handle to this actor instance.
    head : RayActorHandle
        Handle to the :class:`SimulationHead` actor.
    arrays : AsyncDict[str, Array]
        Dictionary mapping array names to their :class:`Array` instances,
        which track chunks and timesteps.
    feedback_non_chunked : dict
        Dictionary for storing non-chunked feedback values shared between
        analytics and simulation.
    """

    async def __init__(self, actor_id: int, arrays_metadata: Dict[str, Dict] = {}) -> None:
        """
        Initialize the base node actor, register with the head, and prime state.

        Parameters
        ----------
        actor_id : int
            Unique identifier for this node actor (typically the node ID).
        arrays_metadata : dict, optional
            Reserved for future use; currently unused. Default is ``{}``.
        """
        self.actor_id = actor_id
        self.actor_handle = ray.get_runtime_context().current_actor

        self.head = get_ready_actor_with_retry(name="simulation_head", namespace="deisa_ray")
        await self.head.register_scheduling_actor.remote(actor_id, self.actor_handle)

        # Keeps track of array metadata AND ref per timestep.
        # TODO: I think these two responsabilities could be separated.
        self.partial_arrays: AsyncDict[str, PartialArray] = AsyncDict()

        # For non-chunked feedback between analytics and simulation
        self.feedback_non_chunked: Dict[Hashable, Any] = {}

    def set(self, *args, key: Hashable, value: Any, chunked: bool = False, **kwargs) -> None:
        """
        Store a feedback value shared between analytics and simulation.

        Parameters
        ----------
        key : Hashable
            Identifier for the feedback value.
        value : Any
            Value to store.
        chunked : bool, optional
            Placeholder for future chunked feedback support. Must remain
            ``False`` today. Default is ``False``.

        Notes
        -----
        The ``*args`` and ``**kwargs`` parameters are accepted for forward
        compatibility with future signatures but are currently unused.
        """
        if not chunked:
            self.feedback_non_chunked[key] = value
        else:
            # TODO: implement chunked version
            raise NotImplementedError()

    def get(
        self,
        key,
        default=None,
        chunked=False,
    ) -> Any:
        """
        Retrieve a feedback value previously stored with :meth:`set`.

        Parameters
        ----------
        key : Hashable
            Identifier of the requested value.
        default : Any, optional
            Value returned when ``key`` is not present. Default is ``None``.
        chunked : bool, optional
            Placeholder for chunked feedback retrieval. Must remain
            ``False`` today. Default is ``False``.

        Returns
        -------
        Any
            Stored value or ``default`` if missing.
        """
        if not chunked:
            return self.feedback_non_chunked.get(key, default)
        else:
            raise NotImplementedError()

    def delete(
        self,
        *args,
        key: Hashable,
        **kwargs,
    ) -> None:
        """
        Delete a feedback value if present.

        Parameters
        ----------
        key : Hashable
            Identifier to remove from the non-chunked feedback store.

        Notes
        -----
        Missing keys are ignored to keep the call idempotent.
        """
        self.feedback_non_chunked.pop(key, None)

    def _create_or_retrieve_partial_array(self, array_name: str, nb_chunks_of_node: int) -> PartialArray:
        """
        Return the partial-array container for ``array_name``, creating it if missing.

        Parameters
        ----------
        array_name : str
            Name of the array being assembled on this node.
        nb_chunks_of_node : int
            Number of chunks contributed by this node. Stored for sanity checks
            when registering metadata.

        Returns
        -------
        PartialArray
            Container tracking metadata and per-timestep chunks.
        """
        if array_name not in self.partial_arrays:
            self.partial_arrays[array_name] = PartialArray()
            self.nb_chunks_of_node = nb_chunks_of_node
        return self.partial_arrays[array_name]

    async def register_chunk_meta(
        self,
        bridge_id: int,
        array_name: str,
        chunk_shape,
        nb_chunks_per_dim,
        nb_chunks_of_node: int,
        dtype,
        chunk_position,
    ) -> None:
        """
        Register chunk metadata contributed by a bridge on this node.

        Parameters
        ----------
        bridge_id : int
            Identifier of the bridge sending the chunk.
        array_name : str
            Name of the array being populated.
        chunk_shape : tuple[int, ...]
            Shape of the chunk owned by this bridge.
        nb_chunks_per_dim : tuple[int, ...]
            Number of chunks per dimension in the global decomposition.
        nb_chunks_of_node : int
            Number of chunks contributed by this node for the array.
        dtype : Any
            NumPy dtype of the chunk.
        chunk_position : tuple[int, ...]
            Position of this chunk in the global grid.

        Notes
        -----
        Once all bridges on the node have registered, the node actor forwards
        the consolidated metadata to the head actor and signals readiness
        for this array. Subsequent callers will await the per-array event.
        """
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
        **kwargs,
    ) -> None:
        """
        Add a chunk of data to this node actor.

        This method is called by Bridge instances to send chunks of arrays
        to this node actor. When all chunks from a node are received,
        the actor notifies the head node that chunks are ready.

        Parameters
        ----------
        bridge_id : int
            Identifier of the bridge that owns this chunk.
        array_name : str
            Name of the array receiving the chunk.
        chunk_ref : list[ray.ObjectRef]
            Single-element list containing the Ray ObjectRef to the chunk
            data. The extra list level is kept for Dask compatibility.
        timestep : int
            Timestep index the chunk belongs to.
        chunked : bool, optional
            Reserved for future multi-chunk sends. Must remain ``True`` for
            the current workflow. Default is ``True``.

        Returns
        -------
        None

        Raises
        ------
        ContractError
            If the array has not been registered via
            :meth:`register_chunk_meta` before chunks are added.

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
            raise ContractError(
                f"User requested to add chunk for {array_name} but this array has"
                f"not been described. Please call register_array({array_name}) before calling"
                "add_chunk()."
            )
        partial_array = self.partial_arrays[array_name]

        if timestep not in partial_array.per_timestep_arrays.keys():
            partial_array.per_timestep_arrays[timestep] = ArrayPerTimestep()
        array_timestep = partial_array.per_timestep_arrays[timestep]

        # in this function, chunk_ref is a [rayref]. Doing ray.get(chunk_ref[0]) would get the data.
        # TODO why?
        # the pack function returns chunk_ref[0] (the rayref to actual data). Note, I don't call it as self._pack_object_ref
        # but rather, call it through the actor handle (so its a remote call). It memans that the local_chunk is a ref, s.t
        # ray.get(ref) returns ANOTHER ref to the data. So, a ref of a ref.
        array_timestep.local_chunks[bridge_id] = self.actor_handle._pack_object_ref.remote(chunk_ref)

        if len(array_timestep.local_chunks) == self.nb_chunks_of_node:
            chunks = []
            for bridge_id, ref in array_timestep.local_chunks._data.items():
                assert isinstance(ref, ray.ObjectRef)
                chunks.append(ref)
                array_timestep.local_chunks[bridge_id] = pickle.dumps(ref)
            # all_chunks_ref is a ref to a list of refs of refs.
            all_chunks_ref = ray.put(chunks)

            # TODO rename
            await self.head.chunks_ready.options(enable_task_events=False).remote(
                array_name, timestep, [all_chunks_ref]
            )
            array_timestep.chunks_ready_event.set()
            array_timestep.chunks_ready_event.clear()
        else:
            await array_timestep.chunks_ready_event.wait()


@ray.remote
class SchedulingActor(NodeActorBase):
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
    scheduling_actors : list[RayActorHandle]
        List of all scheduling actor handles, populated lazily when first
        graph is scheduled.
    graph_infos : AsyncDict[int, GraphInfo]
        Dictionary mapping graph IDs to their :class:`GraphInfo` objects,
        which track graph scheduling state and results.
    """

    async def __init__(self, actor_id: int, arrays_metadata: Dict[str, Dict] = {}) -> None:
        """
        Initialize a scheduling actor with shared node functionality.

        Parameters
        ----------
        actor_id : int
            Unique identifier for this scheduling actor (node ID).
        arrays_metadata : dict, optional
            Reserved for future extensions requiring array metadata at
            construction time. Default is ``{}``.
        """
        # Delegate initialization to NodeActor, which sets up head node
        # registration, arrays, and feedback mechanisms.
        await super().__init__(actor_id=actor_id, arrays_metadata=arrays_metadata)

        # Scheduling-specific state (not needed for plain NodeActor)
        self.scheduling_actors: dict[ActorID, RayActorHandle] = {}
        self.graph_infos: AsyncDict[int, GraphInfo] = AsyncDict()

    async def schedule_graph(self, graph_id: int, graph: dict[GraphKey, GraphValue]) -> None:
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
        graph : dict
            The Dask task graph dictionary. Keys are task identifiers, and
            values may be:
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
        # TODO: I tried moving this in init above, but tests fail, not sure why
        # Find the scheduling actors (lazy initialization)
        if not self.scheduling_actors:
            self.scheduling_actors = await self.head.list_scheduling_actors.options(enable_task_events=False).remote()

        # Create and store graph info for tracking this graph's execution
        info = GraphInfo()
        self.graph_infos[graph_id] = info

        await self.substitute_graph_values_with_refs(graph_id, graph)

        # we need to get result of all these keys to resolve task graph
        keys_needed = list(graph.keys())

        # NOTE: How does the line below finish?
        # Since remote_ray_dask_get returns refs, this function immediately returns and we can proceed with populating the info
        # graph and setting the event. Once the event is set, the get_value rpcs, can resolve (later).

        # NOTE: remote_ray_dask_get.remote() -> ref, we do `await ref` which returns the result
        # of the function, which is a tuple of doubleRef to results (per key). The reason its a double ref
        # is that patched_dask_task_wrapper calls itself remotely. So:
        # patched_dask_task_wrapper(doubleRef) returns patched_dask_task_wrapper.remote(singleRef)
        # but the second time its called as a remote function, it returns a value.
        # so, the function returns a ref -> result. But, since we set ray_persist, we get a ref to the output of the function.        # function. Therefore, we get a ref -> ref -> result.
        # Incindentally, this is why, removing ray_persist = True from remote_ray_dask_get makes everything fail (because we then
        # get a tuple of single refs instead of double). We need double refs to keep the entire graph consistent.
        doubleRefs_of_results: tuple[DoubleRef] = await remote_ray_dask_get.remote(graph, keys_needed)

        # store the refs in a dictionary so other actors can retrieve them
        for key, ref in zip(keys_needed, doubleRefs_of_results):
            info.refs[key] = ref

        info.scheduled_event.set()

    async def substitute_graph_values_with_refs(self, graph_id: int, graph: dict[GraphKey, GraphValue]):
        """
        Replace placeholders in a graph with concrete Ray ObjectRefs.

        Parameters
        ----------
        graph_id : int
            Identifier of the graph being processed.
        graph : dict[GraphKey, GraphValue]
            Task graph that may contain :class:`ChunkRef` or
            :class:`ScheduledByOtherActor` placeholders.

        Notes
        -----
        - ``ScheduledByOtherActor`` entries are rewritten to remote calls
          to the owning scheduling actor.
        - ``ChunkRef`` entries are replaced with the pickled or direct
          ObjectRefs stored locally for the relevant timestep/position.
        - When a stored ref is still in-memory, it is pickled to ensure
          ownership transfer and memory release after scheduling.
        """
        for key, val in graph.items():
            # Adapt external keys
            if isinstance(val, ScheduledByOtherActor):
                actor = self.scheduling_actors[val.actor_id]
                graph[key] = actor.get_value.options(enable_task_events=False).remote(graph_id, key)

            # Replace the false chunks by the real ObjectRefs
            if isinstance(val, ChunkRef):
                assert val.actor_id == self.actor_id

                # TODO: maybe awaiting is not necessary. When would the key not be present in the
                # AsyncDict?
                array = await self.partial_arrays.wait_for_key(val.array_name)

                array_timestep = await array.per_timestep_arrays.wait_for_key(val.timestep)

                # should be pickled ref of ref
                ref = await array_timestep.local_chunks.wait_for_key(val.bridge_id)

                # TODO what does this mean?
                if isinstance(ref, bytes):  # This may not be the case depending on the asyncio scheduling order
                    # technically, here its an actual ray ref which is a ref of ref.
                    ref = pickle.loads(ref)
                else:
                    ref = pickle.loads(pickle.dumps(ref))  # To free the memory automatically

                # replace ChunkRef by actual ref (still ref of ref)
                graph[key] = ref

    # this function does a 1 level unpacking of a ref of ref among other things.
    # TODO rename this function to something better. It is called by other scheduling actors to retrieve
    # a ref for a dask task that this actor posseses (is supposed to deal with). It does three things
    # 1. waits for the graph to be created (look at method above)
    # 2. waits for the scheduled_event to be set
    # 3. retrieves the ref corresponding to the required key from a dictionary. Since this is a ref of ref,
    # it awaits it to unpack one level and return a ref to a result of the task.
    async def get_value(self, graph_id: int, key: str) -> ray.ObjectRef:
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
        # For tomorrow: I understood this - the await is similar to a ray.get() so it unpacks the ref once.
        # therefore, at the end all the refs are refs of refs (same leve). Then the patches dask task wrapper
        # is called and works the same way for all tasks (calls itself).
        # Because of this, I also think I understand why we need a ref of ref: if you work directly with a ref of
        # data, then Actor1 could need a key from Actor2, and Actor2 a key from Actor1. Both call ray.get(refOwnedByOtherActor)
        # and the cluster deadlocks. To fix this, I need to make it non-blocking. How can I do this? By making it a remote call.
        # However, to make the entire graph "coherent", I need to make leaf nodes refs of refs as well. Then it all becomes
        # cohesive.
        # I am missing why we need the pickling and how memory is released.
        double_ref = graph_info.refs[key]
        ref_to_result = await double_ref
        return ref_to_result


@ray.remote
class NodeActor(NodeActorBase):
    """
    Actor responsible for gathering chunks and exchanging data with analytics.

    This is a Ray actor. Shared logic is implemented in NodeBase.
    """

    async def __init__(self, actor_id: int, arrays_metadata: Dict[str, Dict] = {}) -> None:
        """
        Initialize a plain node actor (without scheduling responsibilities).

        Parameters
        ----------
        actor_id : int
            Unique identifier for the node actor.
        arrays_metadata : dict, optional
            Reserved for future metadata consumption. Default is ``{}``.
        """
        # Initialise the shared base part
        await NodeActorBase.__init__(self, actor_id=actor_id, arrays_metadata=arrays_metadata)
        # Optionally: NodeActor-specific init here
