from dataclasses import dataclass
from typing import Callable, Any, TypeAlias
from dask.highlevelgraph import HighLevelGraph
import math
import numpy as np
import asyncio
import ray
import ray.actor
from deisa.ray import Timestep
import dask.array as da
from deisa.ray._async_dict import AsyncDict
from collections import defaultdict

type DoubleRef = ray.ObjectRef
type ActorID = str
# anything used in dask to reprsent a task key (usually its a tuple)
type GraphKey = Any
# GraphValue can be any of: ChunkRef, ScheduledByOtherActor, or anything used in Dask to represent a task value.
type GraphValue = Any
RayActorHandle: TypeAlias = ray.actor.ActorHandle


class ArrayPerTimestep:
    """
    Internal class tracking chunks for a specific array and timestep.

    Tracks the per-timestep chunks owned by a single scheduling actor. Each
    instance is keyed by ``timestep`` inside a :class:`PartialArray`.

    Attributes
    ----------
    chunks_ready_event : asyncio.Event
        Triggered when all chunks for this timestep owned by the node actor
        have arrived and been forwarded to the head actor.
    local_chunks : AsyncDict[int, ray.ObjectRef | bytes]
        Mapping of ``bridge_id`` to the double ObjectRef for that chunk. Once
        forwarded to the head actor the value is replaced with pickled bytes
        to free memory.
    """

    def __init__(self):
        """Create the readiness event and the async storage for local chunks."""
        # Triggered when all the chunks are ready
        self.chunks_ready_event: asyncio.Event = asyncio.Event()

        # {bridgeID: chunk}
        self.local_chunks: AsyncDict[int, ray.ObjectRef | bytes] = AsyncDict()


class PartialArray:
    """
    Internal class tracking metadata and chunks for an array.

    Maintains metadata for a single array on one scheduling actor, including
    the set of locally owned chunks and per-timestep chunk collections.

    Attributes
    ----------
    ready_event : asyncio.Event
        Set once all local bridges have provided metadata for the array and
        the head actor has been notified.
    chunks_contained_meta : set[tuple[int, tuple[int, ...], tuple[int, ...]]]
        Metadata tuples ``(bridge_id, chunk_position, chunk_size)`` for the
        chunks owned by this actor.
    bid_to_pos : dict[int, tuple]
        Maps ``bridge_id`` to the corresponding chunk position for quick
        lookup when forwarding payloads.
    per_timestep_arrays : AsyncDict[Timestep, ArrayPerTimestep]
        Per-timestep structures that hold chunk references as they arrive.
    """

    # TODO add types

    def __init__(self):
        """Initialise per-array metadata containers and readiness flag."""
        # Indicates if register_partial_array method has been called for this array.
        self.ready_event = asyncio.Event()

        # Chunks owned by this actor for this array.
        # {(bridge_id, chunk position, chunk size), ...}
        self.chunks_contained_meta: set[tuple[int, tuple[int, ...], tuple[int, ...]]] = set()
        self.bid_to_pos: dict[int, tuple]= {}

        self.per_timestep_arrays: AsyncDict[Timestep, ArrayPerTimestep] = AsyncDict()


@dataclass
class ScheduledByOtherActor:
    """
    Represents a task that is scheduled by another actor.

    This class is used as a placeholder in Dask task graphs to indicate
    that a task should be scheduled by a different actor. When a task graph
    is sent to an actor, tasks marked with this class will be delegated to
    the specified actor.

    Parameters
    ----------
    actor_id : int
        The ID of the scheduling actor that should schedule this task.

    Notes
    -----
    This is used to handle cross-actor task dependencies in distributed
    Dask computations where different parts of the task graph are handled
    by different scheduling actors.
    """

    actor_id: ActorID


class GraphInfo:
    """
    Information about graphs and their scheduling.

    Tracks scheduling status and produced references for a Dask task graph
    scheduled by a :class:`SchedulingActor`.

    Attributes
    ----------
    scheduled_event : asyncio.Event
        Event set once the graph has been submitted to Ray.
    refs : dict[str, ray.ObjectRef]
        Mapping from task key to the *double* Ray ObjectRef returned by the
        patched Dask-on-Ray scheduler.
    """

    def __init__(self):
        """Create the scheduling event and storage for result references."""
        self.scheduled_event = asyncio.Event()
        self.refs: dict[str, ray.ObjectRef] = {}


@dataclass
class ChunkRef:
    """
    Represents a chunk of an array in a Dask task graph.

    This class is used as a placeholder in Dask task graphs to represent a
    chunk of data. The task corresponding to this object must be scheduled
    by the actor who has the actual data. This class is used since Dask
    tends to inline simple tuples, which would prevent proper scheduling.

    Parameters
    ----------
    actor_id : int
        The ID of the scheduling actor that owns this chunk.
    array_name : str
        The real name of the array, without the timestep suffix.
    timestep : Timestep
        The timestep this chunk belongs to.
    position : tuple[int, ...]
        The position of the chunk in the array decomposition.
    _all_chunks : ray.ObjectRef or None, optional
        ObjectRef containing all chunks for this timestep. Set for one chunk
        only to avoid duplication. Default is None.

    Notes
    -----
    This class is used to prevent Dask from inlining simple tuples in the
    task graph, which would break the scheduling mechanism. The behavior
    may change in newer versions of Dask.
    """

    actor_id: int
    array_name: str  # The real name, without the timestep
    timestep: Timestep
    position: tuple[int, ...]
    bridge_id: int

    # Set for one chunk only.
    _all_chunks: ray.ObjectRef | None = None


@dataclass
class WindowArrayDefinition:
    """
    Description of an array with optional windowing support.

    Parameters
    ----------
    name : str
        The name of the array.
    window_size : int or None, optional
        If specified, creates a sliding window of arrays for this array name.
        The window will contain the last `window_size` timesteps. If None,
        only the current timestep array is provided. Default is None.
    preprocess : Callable, optional
        A preprocessing function to apply to chunks of this array before
        they are sent to the analytics. The function should take a numpy
        array and return a processed numpy array. Default is the identity
        function (no preprocessing).

    Examples
    --------
    >>> def normalize(arr):
    ...     return arr / arr.max()
    >>> # Array with windowing: last 5 timesteps
    >>> array_def = ArrayDefinition(name="temperature", window_size=5, preprocess=normalize)
    >>> # Array without windowing: current timestep only
    >>> array_def = ArrayDefinition(name="pressure", window_size=None)
    """

    name: str
    window_size: int | None = None
    preprocess: Callable = lambda x: x


@dataclass
class _CallbackConfig:
    simulation_callback: Callable
    arrays_description: list[WindowArrayDefinition]
    max_iterations: int
    prepare_iteration: Callable | None
    preparation_advance: int


class DaskArrayData:
    """
    Information about a Dask array being built.

    Tracks metadata and per-timestep state for a Dask array assembled from
    chunks sent by scheduling actors.

    Parameters
    ----------
    name : str
        Array name registered with the head actor.
    f_preprocessing : Callable
        Preprocessing callback applied to each chunk.

    Attributes
    ----------
    name : str
        Array name without timestep suffix.
    f_preprocessing : Callable
        Preprocessing callback supplied at registration.
    fully_defined : asyncio.Event
        Set when every chunk owner has been registered.
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
    position_to_node_actorID : dict[tuple[int, ...], int]
        Mapping from chunk position to the scheduling actor responsible
        for that chunk.
    position_to_bridgeID : dict[tuple, int]
        Mapping from chunk position to the producing bridge ID.
    nb_scheduling_actors : int or None
        Number of unique scheduling actors owning chunks of this array.
        Set when all chunk owners are known.
    chunk_refs : dict[Timestep, list[ray.ObjectRef]]
        For each timestep, the list of per-actor references that keep
        chunk payloads alive in the object store.
    pos_to_ref_by_timestep : defaultdict
        For each timestep, the (position, ref) pairs provided by scheduling
        actors. Used when distributed scheduling is disabled.
    """

    def __init__(self, name, f_preprocessing) -> None:
        """
        Initialise per-array metadata containers.

        Parameters
        ----------
        name : str
            Array name as registered with the head actor.
        f_preprocessing : Callable
            Preprocessing callback applied to each chunk before analytics
            consume the array.
        """
        self.name = name
        self.f_preprocessing = f_preprocessing

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
        self.position_to_node_actorID: dict[tuple[int, ...], int] = {}
        self.position_to_bridgeID: dict[tuple, int] = {}

        # Number of scheduling actors owning chunks of this array.
        self.nb_scheduling_actors: int | None = None

        # Each reference comes from one scheduling actor. The reference a list of
        # ObjectRefs, each ObjectRef corresponding to a chunk. These references
        # shouldn't be used directly. They exists only to release the memory
        # automatically.
        # When the array is buit, these references are put in the object store, and the
        # global reference is added to the Dask graph. Then, the list is cleared.
        self.chunk_refs: dict[Timestep, list[ray.ObjectRef]] = {}

        self.pos_to_ref_by_timestep = defaultdict(list)

    def update_meta(
        self,
        nb_chunks_per_dim: tuple[int, ...],
        dtype: np.dtype,
        position: tuple[int, ...],
        size: tuple[int, ...],
        node_actor_id: int,
        bridge_id: int,
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
        node_actor_id : int
            Scheduling actor that owns this chunk.
        bridge_id : int
            Bridge identifier that produced the chunk (used for lookups).

        Raises
        ------
        AssertionError
            If the chunk position is out of bounds, or if subsequent chunks
            have inconsistent dimensions, dtype, or sizes compared to the
            first chunk.
        """
        # TODO should be done just once
        if self.nb_chunks_per_dim is None:
            self.nb_chunks_per_dim = nb_chunks_per_dim
            self.nb_chunks = math.prod(nb_chunks_per_dim)

            self.dtype = dtype
            self.chunks_size = [[None for _ in range(n)] for n in nb_chunks_per_dim]
        else:
            assert self.nb_chunks_per_dim == nb_chunks_per_dim
            assert self.dtype == dtype
            assert self.chunks_size is not None

        # TODO this actually should be done each time
        self.position_to_node_actorID[position] = node_actor_id
        self.position_to_bridgeID[position] = bridge_id
        for i, pos in enumerate(position):
            if self.chunks_size[i][pos] is None:
                self.chunks_size[i][pos] = size[pos]
            else:
                assert self.chunks_size[i][pos] == size[pos]

    def add_chunk_ref(self, chunk_ref: ray.ObjectRef, timestep: Timestep, pos_to_ref) -> bool:
        """
        Add a reference sent by a scheduling actor.

        Parameters
        ----------
        chunk_ref : ray.ObjectRef
            Ray object reference to a chunk sent by a scheduling actor.
        timestep : Timestep
            The timestep this chunk belongs to.
        pos_to_ref : dict[tuple, ray.ObjectRef]
            Mapping of chunk position to the (double) Ray ObjectRef provided
            by the scheduling actor for this timestep.

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
        self.pos_to_ref_by_timestep[timestep] += list(pos_to_ref.items())

        # We don't know all the owners yet
        # TODO this method is useless now because we no longer "send" this data at every timestep.
        # before it was needed since the nb_chunks could in theory change and we were giving info about
        # array as well. This is no longer the case. Can someone confirm?
        if len(self.position_to_node_actorID) != self.nb_chunks:
            return False

        # done once only and then never again - move it somewhere else?
        if self.nb_scheduling_actors is None:
            self.nb_scheduling_actors = len(set(self.position_to_node_actorID.values()))

        # each actor produces a single ref - once I have as many refs as scheduling actors,
        # I mark the array as ready to be formed.
        return len(self.chunk_refs[timestep]) == self.nb_scheduling_actors

    def get_full_array(self, timestep: Timestep, *, distributing_scheduling_enabled: bool,  is_preparation: bool = False) -> da.Array:
        """
        Return the full Dask array for a given timestep.

        Parameters
        ----------
        timestep : Timestep
            The timestep for which the full array should be returned.
        distributing_scheduling_enabled : bool
            When ``True``, emit a graph containing :class:`ChunkRef` tasks
            for distributed scheduling. When ``False``, materialise the
            actual chunk payloads and build a local Dask array.
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
        When distributed scheduling is enabled the graph uses :class:`ChunkRef`
        placeholders that keep data owner information. Otherwise the concrete
        chunk payloads are inlined. Chunk reference lists are deleted after
        embedding in the graph to avoid leaking memory. ``is_preparation`` skips
        storing payload refs entirely so analytics can inspect shapes/chunks
        without materialising data.
        """
        assert len(self.position_to_node_actorID) == self.nb_chunks
        assert self.nb_chunks is not None and self.nb_chunks_per_dim is not None

        if is_preparation:
            all_chunks = None
        else:
            all_chunks = ray.put(self.chunk_refs[timestep])
            del self.chunk_refs[timestep]

        # We need to add the timestep since the same name can be used several times for different
        # timesteps
        dask_name = f"{self.name}_{timestep}"

        if distributing_scheduling_enabled:
            graph = {
                # We need to repeat the name and position in the value since the key might be removed
                # by the Dask optimizer
                (dask_name,)
                + position: ChunkRef(  # note only first ChunkRef instance contains actual refs, the others contain only metadata.
                    actor_id,
                    self.name,
                    timestep,
                    position,
                    self.position_to_bridgeID[position],
                    _all_chunks=all_chunks if it == 0 else None,
                )
                for it, (position, actor_id) in enumerate(self.position_to_node_actorID.items())
            }
        else:
            # TODO: this could be an antipattern (calling ray.get in for loop). Could maybe put all the 
            # double refs in a list and call ray.wait() or ray.get() on the list?
            # something that submits all the refs one go. 
            graph = {
                (dask_name,)
                + position: ray.get(dr)
                for position, dr in self.pos_to_ref_by_timestep[timestep]
            }

        # Needed for prepare iteration otherwise key lookup fails since iteration does not yet exist
        # TODO ensure flow is as expected
        self.pos_to_ref_by_timestep.pop(timestep, None)

        dsk = HighLevelGraph.from_collections(dask_name, graph, dependencies=())

        full_array = da.Array(
            dsk,
            dask_name,
            chunks=self.chunks_size,
            dtype=self.dtype,
        )

        return full_array
