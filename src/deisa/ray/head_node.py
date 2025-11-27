import asyncio
import logging
import math
from dataclasses import dataclass
from typing import Callable

import dask
import dask.array as da
import numpy as np
import ray
import ray.actor
from dask.highlevelgraph import HighLevelGraph
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from deisa.ray import Timestep
from deisa.ray._scheduler import deisa_ray_get
from deisa.ray._scheduling_actor import ChunkRef


def init():
    """
    Initialize Ray and configure Dask to use the Deisa-Ray scheduler.

    This function initializes Ray if it hasn't been initialized yet, and
    configures Dask to use the Deisa-Ray custom scheduler with task-based
    shuffling.

    Notes
    -----
    Ray is initialized with automatic address detection, logging to driver
    disabled, and error-level logging. Dask is configured to use the
    `deisa_ray_get` scheduler with task-based shuffling.
    """
    if not ray.is_initialized():
        ray.init(address="auto", log_to_driver=False, logging_level=logging.ERROR)

    dask.config.set(scheduler=deisa_ray_get, shuffle="tasks")


@dataclass
class ArrayDefinition:
    """
    Description of a Dask array given by the user.

    Parameters
    ----------
    name : str
        The name of the array.
    preprocess : Callable, optional
        A preprocessing function to apply to chunks of this array before
        they are sent to the analytics. The function should take a numpy
        array and return a processed numpy array. Default is the identity
        function (no preprocessing).

    Examples
    --------
    >>> def normalize(arr):
    ...     return arr / arr.max()
    >>> array_def = ArrayDefinition(name="temperature", preprocess=normalize)
    """

    name: str
    preprocess: Callable = lambda x: x


class _DaskArrayData:
    """
    Information about a Dask array being built.

    This class tracks the metadata and state of a Dask array as it is
    constructed from chunks sent by scheduling actors.

    Parameters
    ----------
    definition : ArrayDefinition
        The definition of the array, including its name and preprocessing
        function.

    Attributes
    ----------
    definition : ArrayDefinition
        The array definition.
    fully_defined : asyncio.Event
        Event that is set when all chunk owners are known.
    nb_chunks_per_dim : tuple[int, ...] or None
        Number of chunks per dimension. Set when first chunk owner is
        registered.
    nb_chunks : int or None
        Total number of chunks in the array. Set when first chunk owner
        is registered.
    chunks_size : list[list[int | None]] or None
        For each dimension, the size of chunks in that dimension. None
        values indicate unknown chunk sizes.
    dtype : np.dtype or None
        The numpy dtype of the array chunks. Set when first chunk owner
        is registered.
    scheduling_actors_id : dict[tuple[int, ...], int]
        Mapping from chunk position to the ID of the scheduling actor
        responsible for that chunk.
    nb_scheduling_actors : int or None
        Number of unique scheduling actors owning chunks of this array.
        Set when all chunk owners are known.
    chunk_refs : dict[Timestep, list[ray.ObjectRef]]
        For each timestep, a list of Ray object references to the chunks.
        These references are used to keep chunks in memory and are cleared
        when the array is built.
    """

    def __init__(self, definition: ArrayDefinition) -> None:
        self.definition = definition

        # This will be set when we know, for each chunk, the scheduling actor in charge of it.
        self.fully_defined: asyncio.Event = asyncio.Event()

        # This will be set when the first chunk is added
        self.nb_chunks_per_dim: tuple[int, ...] | None = None
        self.nb_chunks: int | None = None

        # For each dimension, the size of the chunks in this dimension
        self.chunks_size: list[list[int | None]] | None = None

        # Type of the numpy arrays
        self.dtype: np.dtype | None = None

        # ID of the scheduling actor in charge of the chunk at each position
        self.scheduling_actors_id: dict[tuple[int, ...], int] = {}

        # Number of scheduling actors owning chunks of this array.
        self.nb_scheduling_actors: int | None = None

        # Each reference comes from one scheduling actor. The reference a list of
        # ObjectRefs, each ObjectRef corresponding to a chunk. These references
        # shouldn't be used directly. They exists only to release the memory
        # automatically.
        # When the array is buit, these references are put in the object store, and the
        # global reference is added to the Dask graph. Then, the list is cleared.
        self.chunk_refs: dict[Timestep, list[ray.ObjectRef]] = {}

    def set_chunk_owner(
        self,
        nb_chunks_per_dim: tuple[int, ...],
        dtype: np.dtype,
        position: tuple[int, ...],
        size: tuple[int, ...],
        scheduling_actor_id: int,
    ) -> None:
        """
        Register a scheduling actor as the owner of a chunk at a specific position.

        This method records which scheduling actor is responsible for a chunk
        and updates the array metadata. If this is the first chunk registered,
        it initializes the array dimensions and dtype.

        Parameters
        ----------
        nb_chunks_per_dim : tuple[int, ...]
            Number of chunks per dimension in the array decomposition.
        dtype : np.dtype
            The numpy dtype of the chunk.
        position : tuple[int, ...]
            The position of the chunk in the array decomposition.
        size : tuple[int, ...]
            The size of the chunk along each dimension.
        scheduling_actor_id : int
            The ID of the scheduling actor that owns this chunk.

        Raises
        ------
        AssertionError
            If the chunk position is out of bounds, or if subsequent chunks
            have inconsistent dimensions, dtype, or sizes compared to the
            first chunk.
        """
        if self.nb_chunks_per_dim is None:
            self.nb_chunks_per_dim = nb_chunks_per_dim
            self.nb_chunks = math.prod(nb_chunks_per_dim)

            self.dtype = dtype
            self.chunks_size = [[None for _ in range(n)] for n in nb_chunks_per_dim]
        else:
            assert self.nb_chunks_per_dim == nb_chunks_per_dim
            assert self.dtype == dtype
            assert self.chunks_size is not None

        for pos, nb_chunks in zip(position, nb_chunks_per_dim):
            assert 0 <= pos < nb_chunks

        self.scheduling_actors_id[position] = scheduling_actor_id

        for d in range(len(position)):
            if self.chunks_size[d][position[d]] is None:
                self.chunks_size[d][position[d]] = size[d]
            else:
                assert self.chunks_size[d][position[d]] == size[d]

    def add_chunk_ref(self, chunk_ref: ray.ObjectRef, timestep: Timestep) -> bool:
        """
        Add a reference sent by a scheduling actor.

        Parameters
        ----------
        chunk_ref : ray.ObjectRef
            Ray object reference to a chunk sent by a scheduling actor.
        timestep : Timestep
            The timestep this chunk belongs to.

        Returns
        -------
        bool
            True if all chunks for this timestep are ready (i.e., all
            scheduling actors have sent their chunks), False otherwise.

        Notes
        -----
        This method adds the chunk reference to the list for the given
        timestep. It returns True only when all chunk owners are known
        and all have sent their chunks for this timestep.
        """
        self.chunk_refs[timestep].append(chunk_ref)

        # We don't know all the owners yet
        if len(self.scheduling_actors_id) != self.nb_chunks:
            return False

        if self.nb_scheduling_actors is None:
            self.nb_scheduling_actors = len(set(self.scheduling_actors_id.values()))

        return len(self.chunk_refs[timestep]) == self.nb_scheduling_actors

    def get_full_array(self, timestep: Timestep, *, is_preparation: bool = False) -> da.Array:
        """
        Return the full Dask array for a given timestep.

        Parameters
        ----------
        timestep : Timestep
            The timestep for which the full array should be returned.
        is_preparation : bool, optional
            If True, the array will not contain ObjectRefs to the actual data.
            This is used for preparation arrays where only the structure is
            needed. Default is False.

        Returns
        -------
        da.Array
            A Dask array representing the full decomposed array for the given
            timestep. The array is constructed from chunk references stored
            in Ray's object store.

        Raises
        ------
        AssertionError
            If not all chunk owners have been registered, or if the number
            of chunks is inconsistent.

        Notes
        -----
        The array is built using a HighLevelGraph with ChunkRef tasks. When
        `is_preparation` is False, the chunk references are stored in Ray's
        object store and cleared from the local `chunk_refs` dictionary.
        The array name includes the timestep to allow the same array name
        to be used for different timesteps.
        """
        assert len(self.scheduling_actors_id) == self.nb_chunks
        assert self.nb_chunks is not None and self.nb_chunks_per_dim is not None

        if is_preparation:
            all_chunks = None
        else:
            all_chunks = ray.put(self.chunk_refs[timestep])
            del self.chunk_refs[timestep]

        # We need to add the timestep since the same name can be used several times for different
        # timesteps
        dask_name = f"{self.definition.name}_{timestep}"

        graph = {
            # We need to repeat the name and position in the value since the key might be removed
            # by the Dask optimizer
            (dask_name,) + position: ChunkRef(
                actor_id,
                self.definition.name,
                timestep,
                position,
                _all_chunks=all_chunks if it == 0 else None,
            )
            for it, (position, actor_id) in enumerate(self.scheduling_actors_id.items())
        }

        dsk = HighLevelGraph.from_collections(dask_name, graph, dependencies=())

        full_array = da.Array(
            dsk,
            dask_name,
            chunks=self.chunks_size,
            dtype=self.dtype,
        )

        return full_array


def get_head_node_id() -> str:
    """
    Get the node ID of the Ray cluster head node.

    Returns
    -------
    str
        The node ID of the head node.

    Raises
    ------
    AssertionError
        If there is not exactly one head node in the cluster.

    Notes
    -----
    This function queries Ray's state API to find the head node. It assumes
    there is exactly one head node in the cluster.
    """
    from ray.util import state

    nodes = state.list_nodes(filters=[("is_head_node", "=", True)])

    assert len(nodes) == 1, "There should be exactly one head node"

    return nodes[0].node_id


def get_head_actor_options() -> dict:
    """
    Return the options that should be used to start the head actor.

    Returns
    -------
    dict
        Dictionary of Ray actor options including:
        - name: "simulation_head"
        - namespace: "deisa_ray"
        - scheduling_strategy: NodeAffinitySchedulingStrategy for the head node
        - max_concurrency: Very high value to prevent blocking
        - lifetime: "detached" to persist beyond function scope
        - enable_task_events: False for performance

    Notes
    -----
    The head actor is scheduled on the head node with a detached lifetime
    to ensure it persists. High concurrency is set to prevent the actor
    from being blocked when gathering many references.
    """
    return dict(
        # The workers will be able to access to this actor using its name
        name="simulation_head",
        namespace="deisa_ray",
        # Schedule the actor on this node
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=get_head_node_id(),
            soft=False,
        ),
        # Prevents the actor from being stuck when it needs to gather many refs
        max_concurrency=1000_000_000,
        # Prevents the actor from being deleted when the function ends
        lifetime="detached",
        # Disabled for performance reasons
        enable_task_events=False,
    )


@ray.remote
class SimulationHead:
    """
    Ray remote actor that coordinates array construction from simulation chunks.

    The SimulationHead actor manages the construction of Dask arrays from chunks
    sent by scheduling actors. It coordinates multiple scheduling actors,
    tracks array state, and provides arrays to analytics when they are ready.

    Parameters
    ----------
    arrays_definitions : list[ArrayDefinition]
        List of array definitions describing the arrays to be created.
    max_pending_arrays : int, optional
        Maximum number of arrays that can be being built or waiting to be
        collected at the same time. Setting this value can prevent the
        simulation from being many iterations ahead of the analytics.
        Default is 1_000_000_000.

    Attributes
    ----------
    scheduling_actors : dict[str, ray.actor.ActorHandle]
        Mapping from scheduling actor IDs to their actor handles.
    new_pending_array_semaphore : asyncio.Semaphore
        Semaphore used to limit the number of pending arrays.
    new_array_created : asyncio.Event
        Event set when a new array timestep is created.
    arrays : dict[str, _DaskArrayData]
        Mapping from array names to their data structures.
    arrays_ready : asyncio.Queue[tuple[str, Timestep, da.Array]]
        Queue of ready arrays waiting to be collected by analytics.

    Notes
    -----
    This is a Ray remote actor that must be created using
    `SimulationHead.options(**get_head_actor_options()).remote(...)`.
    The actor coordinates between simulation nodes (via scheduling actors)
    and analytics that consume the constructed arrays.
    """

    def __init__(self, arrays_definitions: list[ArrayDefinition], max_pending_arrays: int = 1_000_000_000) -> None:
        # For each ID of a simulation node, the corresponding scheduling actor
        self.scheduling_actors: dict[str, ray.actor.ActorHandle] = {}

        # Must be used before creating a new array, to prevent the simulation from being
        # too many iterations in advance of the analytics.
        self.new_pending_array_semaphore = asyncio.Semaphore(max_pending_arrays)

        self.new_array_created = asyncio.Event()

        self.arrays: dict[str, _DaskArrayData] = {
            definition.name: _DaskArrayData(definition) for definition in arrays_definitions
        }

        # All the newly created arrays
        self.arrays_ready: asyncio.Queue[tuple[str, Timestep, da.Array]] = asyncio.Queue()

    def list_scheduling_actors(self) -> list[ray.actor.ActorHandle]:
        """
        Return the list of scheduling actors.

        Returns
        -------
        list[ray.actor.ActorHandle]
            List of actor handles for all registered scheduling actors.
        """
        return self.scheduling_actors

    async def register_scheduling_actor(self, actor_id: str, actor_handle: ray.actor.ActorHandle):
        """
        Register a scheduling actor that has been created.

        Parameters
        ----------
        actor_id : str
            Unique identifier for the scheduling actor.
        actor_handle : ray.actor.ActorHandle
            Ray actor handle for the scheduling actor.

        Notes
        -----
        If an actor with the same ID is already registered, this method
        does nothing. Scheduling actors register themselves when they are
        created by Bridge instances.
        """
        if actor_id not in self.scheduling_actors:
            self.scheduling_actors[actor_id] = actor_handle

    def preprocessing_callbacks(self) -> dict[str, Callable]:
        """
        Return the preprocessing callbacks for each array.

        Returns
        -------
        dict[str, Callable]
            Dictionary mapping array names to their preprocessing callback
            functions. These callbacks are used by Bridge instances to
            preprocess chunks before sending them to the scheduling actors.

        Notes
        -----
        The preprocessing callbacks are extracted from the array definitions
        provided during initialization. These callbacks are static and cannot
        be changed after initialization.
        """
        return {name: array.definition.preprocess for name, array in self.arrays.items()}

    def set_owned_chunks(
        self,
        scheduling_actor_id: int,
        array_name: str,
        dtype: np.dtype,
        nb_chunks_per_dim: tuple[int, ...],
        chunks: list[tuple[tuple[int, ...], tuple[int, ...]]],  # [(chunk position, chunk size), ...]
    ):
        """
        Register which chunks are owned by a scheduling actor.

        Parameters
        ----------
        scheduling_actor_id : int
            The ID of the scheduling actor that owns these chunks.
        array_name : str
            The name of the array these chunks belong to.
        dtype : np.dtype
            The numpy dtype of the chunks.
        nb_chunks_per_dim : tuple[int, ...]
            Number of chunks per dimension in the array decomposition.
        chunks : list[tuple[tuple[int, ...], tuple[int, ...]]]
            List of tuples, each containing (chunk_position, chunk_size).
            The chunk_position is a tuple of indices, and chunk_size is a
            tuple of sizes along each dimension.

        Notes
        -----
        This method is called by scheduling actors to inform the head actor
        which chunks they are responsible for. This information is used to
        track array construction progress.
        """
        array = self.arrays[array_name]

        for position, size in chunks:
            array.set_chunk_owner(nb_chunks_per_dim, dtype, position, size, scheduling_actor_id)

    async def chunks_ready(self, array_name: str, timestep: Timestep, all_chunks_ref: list[ray.ObjectRef]) -> None:
        """
        Called by scheduling actors to inform the head actor that chunks are ready.

        This method is called when a scheduling actor has prepared all its
        chunks for a given timestep. The head actor collects these chunk
        references and constructs the full Dask array when all chunks from
        all scheduling actors are ready.

        Parameters
        ----------
        array_name : str
            The name of the array these chunks belong to.
        timestep : Timestep
            The timestep these chunks belong to.
        all_chunks_ref : list[ray.ObjectRef]
            List of Ray object references to the chunks. These references
            point to data stored in Ray's object store.

        Notes
        -----
        This method may block if the timestep hasn't been initialized yet.
        It waits for either the semaphore to allow a new pending array or
        for the timestep to be created by another scheduling actor. When all
        chunks for a timestep are ready, the full Dask array is constructed
        and added to the `arrays_ready` queue for collection by analytics.
        """
        array = self.arrays[array_name]

        while timestep not in array.chunk_refs:
            t1 = asyncio.create_task(self.new_pending_array_semaphore.acquire())
            t2 = asyncio.create_task(self.new_array_created.wait())

            done, pending = await asyncio.wait([t1, t2], return_when=asyncio.FIRST_COMPLETED)

            for task in pending:
                task.cancel()

            if t1 in done:
                if timestep in array.chunk_refs:
                    # The array was already created by another scheduling actor
                    self.new_pending_array_semaphore.release()
                else:
                    array.chunk_refs[timestep] = []

                    self.new_array_created.set()
                    self.new_array_created.clear()

        is_ready = array.add_chunk_ref(all_chunks_ref[0], timestep)

        if is_ready:
            self.arrays_ready.put_nowait(
                (
                    array_name,
                    timestep,
                    array.get_full_array(timestep),
                )
            )
            array.fully_defined.set()

    def ready(self):
        """
        Check if the head actor is ready.

        Returns
        -------
        bool
            Always returns True. This method serves as a readiness check
            for the actor.

        Notes
        -----
        This method can be called to verify that the actor has been
        successfully initialized and is ready to receive requests.
        """
        return True

    async def get_next_array(self) -> tuple[str, Timestep, da.Array]:
        """
        Get the next ready array from the queue.

        Returns
        -------
        tuple[str, Timestep, da.Array]
            A tuple containing:
            - array_name: The name of the array
            - timestep: The timestep index
            - array: The full Dask array for this timestep

        Notes
        -----
        This method blocks until an array is available in the `arrays_ready`
        queue. When an array is retrieved, the semaphore is released to
        allow the simulation to create new arrays. This method is typically
        called by analytics to get the next available array for processing.
        """
        array = await self.arrays_ready.get()
        self.new_pending_array_semaphore.release()
        return array

    async def get_preparation_array(self, array_name: str, timestep: Timestep) -> da.Array:
        """
        Return the full Dask array for a given timestep, used for preparation.

        This method returns a Dask array structure without actual data
        references, which is useful for preparation tasks that need to know
        the array structure but don't need the actual data.

        Parameters
        ----------
        array_name : str
            The name of the array.
        timestep : Timestep
            The timestep for which the full array should be returned.

        Returns
        -------
        da.Array
            A Dask array representing the structure of the array for the
            given timestep. This array does not contain ObjectRefs to actual
            data (is_preparation=True).

        Notes
        -----
        This method waits until the array is fully defined (all chunk owners
        are known) before returning. The returned array is suitable for
        preparation tasks that need array metadata but not the actual data.
        """
        await self.arrays[array_name].fully_defined.wait()
        return self.arrays[array_name].get_full_array(timestep, is_preparation=True)
