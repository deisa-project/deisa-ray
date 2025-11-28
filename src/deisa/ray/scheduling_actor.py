import pickle
import numpy as np
import ray
import ray.actor
from deisa.ray._async_dict import AsyncDict
from deisa.ray.types import ChunkRef, ScheduledByOtherActor, GraphInfo, ArrayTimestep, Array
from deisa.ray.utils import get_ready_actor_with_retry
from deisa.ray.ray_patch import remote_ray_dask_get

@ray.remote
class SchedulingActor:
    """
    Actor responsible for gathering chunks and scheduling Dask task graphs.

    Each SchedulingActor is associated with a specific node and is responsible
    for:
    - Collecting chunks of arrays sent by simulation nodes (via Bridge)
    - Registering with the head node
    - Scheduling Dask task graphs that operate on the collected chunks
    - Coordinating with other scheduling actors for cross-actor dependencies

    Parameters
    ----------
    actor_id : int
        Unique identifier for this scheduling actor, typically the node ID.

    Attributes
    ----------
    actor_id : int
        The unique identifier for this scheduling actor.
    actor_handle : ray.actor.ActorHandle
        Handle to this actor instance.
    head : ray.actor.ActorHandle
        Handle to the SimulationHead actor.
    scheduling_actors : list[ray.actor.ActorHandle]
        List of all scheduling actor handles, populated when first graph
        is scheduled.
    arrays : AsyncDict[str, _Array]
        Dictionary mapping array names to their _Array objects, which track
        chunks and timesteps.
    graph_infos : AsyncDict[int, GraphInfo]
        Dictionary mapping graph IDs to their GraphInfo objects, which track
        graph scheduling state and results.

    Notes
    -----
    This is a Ray remote actor that must be created with specific options
    (see Bridge class). The actor registers itself with the head node during
    initialization and waits for chunks to be added via the `add_chunk` method.
    When task graphs are submitted, the actor schedules them and coordinates
    with other actors for distributed execution.
    """

    async def __init__(self, actor_id: int) -> None:
        self.actor_id = actor_id
        self.actor_handle = ray.get_runtime_context().current_actor

        self.head = get_ready_actor_with_retry(name="simulation_head", namespace="deisa_ray")
        await self.head.register_scheduling_actor.remote(actor_id, self.actor_handle)
        self.scheduling_actors: list[ray.actor.ActorHandle] = []

        # For collecting chunks
        self.arrays: AsyncDict[str, Array] = AsyncDict()

        # For scheduling
        self.graph_infos: AsyncDict[int, GraphInfo] = AsyncDict()

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
        them to this scheduling actor.
        """
        # return obect ref
        p_clbs = self.head.preprocessing_callbacks.remote()
        assert isinstance(p_clbs, ray.ObjectRef)
        return p_clbs

    def ready(self) -> None:
        """
        Check if the scheduling actor is ready.

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

    async def add_chunk(
        self,
        array_name: str,
        timestep: int,
        chunk_position: tuple[int, ...],
        dtype: np.dtype,
        nb_chunks_per_dim: tuple[int, ...],
        nb_chunks_of_node: int,
        chunk: list[ray.ObjectRef],
        chunk_shape: tuple[int, ...],
    ) -> None:
        """
        Add a chunk of data to this scheduling actor.

        This method is called by Bridge instances to send chunks of arrays
        to this scheduling actor. When all chunks from a node are received,
        the actor registers the array with the head node (if not already
        registered) and notifies the head node that chunks are ready.

        Parameters
        ----------
        array_name : str
            The name of the array this chunk belongs to.
        timestep : int
            The timestep this chunk belongs to.
        chunk_position : tuple[int, ...]
            The position of the chunk in the array decomposition.
        dtype : np.dtype
            The numpy dtype of the chunk.
        nb_chunks_per_dim : tuple[int, ...]
            Number of chunks per dimension in the array decomposition.
        nb_chunks_of_node : int
            Total number of chunks sent by the node for this timestep.
        chunk : list[ray.ObjectRef]
            List containing a single Ray object reference to the chunk data.
        chunk_shape : tuple[int, ...]
            The shape of the chunk along each dimension.

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
        if array_name not in self.arrays:
            self.arrays[array_name] = Array()
        array = self.arrays[array_name]

        if timestep not in array.timesteps:
            array.timesteps[timestep] = ArrayTimestep()
        array_timestep = array.timesteps[timestep]

        assert chunk_position not in array_timestep.local_chunks
        array_timestep.local_chunks[chunk_position] = self.actor_handle._pack_object_ref.remote(chunk)

        array.owned_chunks.add((chunk_position, chunk_shape))

        if len(array_timestep.local_chunks) == nb_chunks_of_node:
            if not array.is_registered:
                # Register the array with the head node
                await self.head.set_owned_chunks.options(enable_task_events=False).remote(
                    self.actor_id,
                    array_name,
                    dtype,
                    nb_chunks_per_dim,
                    list(array.owned_chunks),
                )
                array.is_registered = True

            chunks = []
            for position, size in array.owned_chunks:
                c = array_timestep.local_chunks[position]
                assert isinstance(c, ray.ObjectRef)
                chunks.append(c)
                array_timestep.local_chunks[position] = pickle.dumps(c)

            all_chunks_ref = ray.put(chunks)

            await self.head.chunks_ready.options(enable_task_events=False).remote(
                array_name, timestep, [all_chunks_ref]
            )

            array_timestep.chunks_ready_event.set()
            array_timestep.chunks_ready_event.clear()
        else:
            await array_timestep.chunks_ready_event.wait()

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
        # Find the scheduling actors
        if not self.scheduling_actors:
            self.scheduling_actors = await self.head.list_scheduling_actors.options(enable_task_events=False).remote()

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

                array = await self.arrays.wait_for_key(val.array_name)
                array_timestep = await array.timesteps.wait_for_key(val.timestep)
                ref = await array_timestep.local_chunks.wait_for_key(val.position)

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
