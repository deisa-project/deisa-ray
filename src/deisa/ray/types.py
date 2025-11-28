from dataclasses import dataclass
from typing import Callable
from deisa.ray._scheduling_actor import ChunkRef
from dask.highlevelgraph import HighLevelGraph
import math
import numpy as np
import asyncio
import ray
from deisa.ray import Timestep
import dask.array as da

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
class HeadArrayDefinition:
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

class DaskArrayData:
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

    def __init__(self, definition: HeadArrayDefinition) -> None:
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
