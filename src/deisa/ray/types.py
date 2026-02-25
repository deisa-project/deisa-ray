import asyncio
from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any, TypeAlias, Literal, Callable

import dask.array as da
from dask.highlevelgraph import HighLevelGraph
import dask.delayed
import numpy as np
import ray
import ray.actor

import pathlib

from deisa.ray import Timestep
from deisa.ray._async_dict import AsyncDict

type DoubleRef = ray.ObjectRef
type ActorID = str
# anything used in dask to represent a task key (usually it's a tuple)
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
        self.bid_to_pos: dict[int, tuple] = {}

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
    ref : ray.ObjectRef
        Ray ObjectRef that eventually points to the chunk data. This is a
        ``ref`` of a ``ref`` produced by the patched Dask scheduler.
    actorid : int
        The ID of the scheduling actor that owns this chunk.
    array_name : str
        The real name of the array, without the timestep suffix.
    timestep : Timestep
        The timestep this chunk belongs to.
    bridge_id : int
        Identifier of the bridge that produced this chunk.

    Notes
    -----
    This class is used to prevent Dask from inlining simple tuples in the
    task graph, which would break the scheduling mechanism. The behavior
    may change in newer versions of Dask.
    """

    ref: ray.ObjectRef
    actorid: int
    array_name: str
    timestep: Timestep
    bridge_id: int


@dataclass
class WindowSpec:
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

    Examples
    --------
    >>> def normalize(arr):
    ...     return arr / arr.max()
    >>> # Array with windowing: last 5 timesteps
    >>> array_def = ArrayDefinition(name="temperature", window_size=5)
    >>> # Array without windowing: current timestep only
    >>> array_def = ArrayDefinition(name="pressure", window_size=None)
    """

    name: str
    window_size: int | None = None


@dataclass(frozen=True)
class DeisaArray:
    dask: da.Array
    t: int

    def to_zarr(self, fname: str, component: str) -> None:
        """
        Save data using the zarr storage format

        Parameters
        ----------
        fname : str
            The name of the zarr storage where the data will be stored.

        component : str
            Component to save in zarr storage

        Notes
        -----
        This method is a simple wrapper to `dask.to_zarr`.
        https://docs.dask.org/en/latest/generated/dask.array.to_zarr.html#dask.array.to_zarr
        """

        full_path = pathlib.Path(fname).expanduser().resolve()
        da.to_zarr(self.dask.persist(), full_path, component=component, compute=True)

    def to_hdf5(self, fname: str, dataset: str) -> None:
        """
        Save data to a HDF5 file (using HDF5 VDS).

        Parameters
        ----------
        fname : str
            The name of the final file where the data will be stored.
        dataset : str
            The name of the dataset in the hdf5 where the data will be stored.

        Notes
        -----
        This method creates files for each chunk, then links the
        files using HDF5 VDS. The chunk file is named as
        `.{filename}-{chunk_position}.h5`.
        """

        to_hdf5(fname, {dataset: self})


def to_hdf5(fname: str, sources: dict[str, DeisaArray]) -> None:
    """
    Save data to a HDF5 file (using HDF5 VDS).

    Parameters
    ----------
    fname : str
        The name of the final file where the data will be stored.
    sources : dict[str, DeisaArray]
        Dict mapping the datasets in final file to final files

    Notes
    -----
    This method creates files for each chunk, then links the
    files using HDF5 VDS. The chunk file is named as
    `.{filename}-{chunk_position}.h5`.
    """

    import h5py

    def chunk_fname(fname: str, dataset: str, chunkid: tuple[int, ...] = ()):
        """
        Create the filename for a chunk.

        Parameters
        ----------
        fname : str
            The name of the final file where the data will be stored.
        dataset : str
            The name of the dataset in the hdf5 where the data will be stored.
        block_id : tuple[int, ...]
            Chunk position to create the file.

        Returns
        -------
        str
            Filename for the chunk of position block_id.
        """

        path = pathlib.Path(fname).expanduser().resolve()
        parents, name, suffix = path.parents[0], path.stem, path.suffix
        chunk_str = "-".join(map(str, chunkid))

        # Hidden name for the chunk files
        new_name = "." + name + "-" + dataset + f"-{chunk_str}" + suffix

        return parents / new_name

    def save_chunk(chunk: np.ndarray, fname: str, dataset: str, block_id: tuple[int, ...] | None = None) -> None:
        """
        Save one chunk to a individual hdf5 file.

        Parameters
        ----------
        chunk : np.ndarray
            Chunk to be stored.
        fname : str
            The name of the final file where the data will be stored.
        dataset : str
            The name of the dataset in the hdf5 where the data will be stored.
        block_id : tuple[int, ...]
            Chunk position, used to merge into a VDS.
        """

        filename = chunk_fname(fname, dataset, block_id)

        with h5py.File(filename, "w") as f:
            f.create_dataset(dataset, data=chunk)

    def create_vds(
        fname: str,
        dataset: str,
        chunk_shape: tuple[int, ...],
        data_shape: tuple[int, ...],
        nb_chunks_per_dim: tuple[int, ...],
        data_dtype: np.dtype,
    ) -> None:
        """
        Creates a VDS aggregating all chunk files.

        Parameters
        ----------
        fname : str
            The name of the final file where the data will be stored.
        dataset : str
            The name of the dataset in the hdf5 where the data will be stored.
        chunk_shape : tuple[int,...]
            Shape of the chunks, used to map the chunks into the VDS.4
        data_shape : tuple[int,...]
            Shape of the data.
        nb_chunks_per_dim : tuple[int,...]
            Number of chunks for each dimension
        data_dtype : np.dtype
            The numpy dtype of the data.
        """

        layout = h5py.VirtualLayout(shape=data_shape, dtype=data_dtype)

        for block_id in np.ndindex(nb_chunks_per_dim):
            name = chunk_fname(fname, dataset, block_id)
            vsource = h5py.VirtualSource(name, dataset, shape=chunk_shape)

            selection = tuple(slice(idx * size, (idx + 1) * size) for idx, size in zip(block_id, chunk_shape))

            layout[selection] = vsource

        with h5py.File(fname, "a", libver="latest") as f:
            f.create_virtual_dataset(dataset, layout, fillvalue=-1)

    full_path = pathlib.Path(fname).expanduser().resolve()

    writing_tasks = []
    for dataset, deisa in sources.items():
        delayed_grid = deisa.dask.to_delayed()

        for block_id in np.ndindex(delayed_grid.shape):
            chunk = delayed_grid[block_id]

            writing_tasks.append(
                dask.delayed(save_chunk)(chunk, fname=str(full_path), dataset=dataset, block_id=block_id)
            )

        create_vds(full_path, dataset, deisa.dask.chunksize, deisa.dask.shape, deisa.dask.numblocks, deisa.dask.dtype)

    dask.compute(*writing_tasks)


@dataclass
class _CallbackConfig:
    simulation_callback: Callable
    arrays_description: list[WindowSpec]
    exception_handler: Callable
    when: Literal["AND", "OR"]


class DaskArrayData:
    """
    Information about a Dask array being built.

    Tracks metadata and per-timestep state for a Dask array assembled from
    chunks sent by scheduling actors.

    Parameters
    ----------
    name : str
        Array name registered with the head actor.

    Attributes
    ----------
    name : str
        Array name without timestep suffix.
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

    def __init__(self, name) -> None:
        """
        Initialise per-array metadata containers.

        Parameters
        ----------
        name : str
            Array name as registered with the head actor.

        """
        self.name = name

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

        self.pos_to_ref_by_timestep: dict[Timestep, list[tuple[tuple, ray.ObjectRef]]] = defaultdict(list)

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
                # NOTE : Verify this
                self.chunks_size[i][pos] = size[i]
            else:
                assert self.chunks_size[i][pos] == size[i]

    def add_chunk_ref(
        self, chunk_ref: ray.ObjectRef, timestep: Timestep, pos_to_ref: dict[tuple, ray.ObjectRef]
    ) -> bool:
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

    def get_full_array(self, timestep: Timestep, *, distributing_scheduling_enabled: bool) -> da.Array:
        """
        Return the full Dask array for a given timestep.

        Parameters
        ----------
        timestep : Timestep
            The timestep for which the full array should be returned.

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
        embedding in the graph to avoid leaking memory.
        """
        assert len(self.position_to_node_actorID) == self.nb_chunks
        assert self.nb_chunks is not None and self.nb_chunks_per_dim is not None

        # We need to add the timestep since the same name can be used several times for different
        # timesteps
        dask_name = f"{self.name}_{timestep}"

        del self.chunk_refs[timestep]

        if distributing_scheduling_enabled:
            graph = {
                (dask_name,) + position: ChunkRef(
                    ref=ref,
                    actorid=self.position_to_node_actorID[
                        next((pos for pos, r in self.pos_to_ref_by_timestep[timestep] if r == ref), None)
                    ],
                    array_name=self.name,
                    timestep=timestep,
                    bridge_id=self.position_to_bridgeID[position],
                )
                for position, ref in self.pos_to_ref_by_timestep[timestep]
            }
        else:
            graph = {(dask_name,) + position: ref for position, ref in self.pos_to_ref_by_timestep[timestep]}

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
