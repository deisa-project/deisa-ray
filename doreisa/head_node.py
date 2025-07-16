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

from doreisa import Timestep
from doreisa._scheduler import doreisa_get
from doreisa._scheduling_actor import ChunkRef, SchedulingActor


def init():
    if not ray.is_initialized():
        ray.init(address="auto", log_to_driver=False, logging_level=logging.ERROR)

    dask.config.set(scheduler=doreisa_get, shuffle="tasks")


@dataclass
class ArrayDefinition:
    """
    Description of a Dask array given by the user.
    """

    name: str
    preprocess: Callable = lambda x: x


class _DaskArrayData:
    """
    Information about a Dask array being built.
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

        # Each reference comes from one scheduling actor. The reference a dictionary of
        # ObjectRefs, each ObjectRef corresponding to a chunk.
        self.chunk_refs: dict[Timestep, dict[int, ray.ObjectRef]] = {}

    def set_chunk_owner(
        self,
        nb_chunks_per_dim: tuple[int, ...],
        dtype: np.dtype,
        position: tuple[int, ...],
        size: tuple[int, ...],
        scheduling_actor_id: int,
    ) -> None:
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

    def add_chunk_ref(self, scheduling_actor_id: int, chunk_ref: ray.ObjectRef, timestep: Timestep) -> bool:
        """
        Add a reference sent by a scheduling actor.

        Return:
            True if all the chunks for this timestep are ready, False otherwise.
        """
        self.chunk_refs[timestep][scheduling_actor_id] = chunk_ref

        # We don't know all the owners yet
        if len(self.scheduling_actors_id) != self.nb_chunks:
            return False

        if self.nb_scheduling_actors is None:
            self.nb_scheduling_actors = len(set(self.scheduling_actors_id.values()))

        return len(self.chunk_refs[timestep]) == self.nb_scheduling_actors

    def get_full_array(self, timestep: Timestep, *, is_preparation: bool = False) -> da.Array:
        """
        Return the full Dask array for a given timestep.

        Args:
            timestep: The timestep for which the full array should be returned.
            is_preparation: If True, the array will not contain ObjectRefs to the
                actual data.
        """
        assert len(self.scheduling_actors_id) == self.nb_chunks
        assert self.nb_chunks is not None and self.nb_chunks_per_dim is not None

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
                all_chunks=None if is_preparation else self.chunk_refs[timestep][actor_id],
            )
            for position, actor_id in self.scheduling_actors_id.items()
        }

        dsk = HighLevelGraph.from_collections(dask_name, graph, dependencies=())

        full_array = da.Array(
            dsk,
            dask_name,
            chunks=self.chunks_size,
            dtype=self.dtype,
        )

        return full_array


def get_head_actor_options() -> dict:
    """Return the options that should be used to start the head actor."""
    return dict(
        # The workers will be able to access to this actor using its name
        name="simulation_head",
        namespace="doreisa",
        # Schedule the actor on this node
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(),
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
    def __init__(self, arrays_definitions: list[ArrayDefinition], max_pending_arrays: int = 1_000_000_000) -> None:
        """
        Initialize the simulation head.

        Args:
            arrays_description: Description of the arrays to be created.
            max_pending_arrays: Maximum number of arrays that can be being built or
                waiting to be collected at the same time. Setting the value can prevent
                the simulation to be many iterations in advance of the analytics.
        """

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
        """
        return list(self.scheduling_actors.values())

    async def scheduling_actor(self, node_id: str, *, is_fake_id: bool = False) -> ray.actor.ActorHandle:
        """
        Return the scheduling actor for the given node ID.

        Args:
            node_id: The ID of the node.
            is_fake_id: If True, the ID isn't a Ray node ID, and the actor can be scheduled
                anywhere. This is useful for testing purposes.
        """

        if node_id not in self.scheduling_actors:
            actor_id = len(self.scheduling_actors)

            if is_fake_id:
                self.scheduling_actors[node_id] = SchedulingActor.remote(actor_id)  # type: ignore
            else:
                self.scheduling_actors[node_id] = SchedulingActor.options(  # type: ignore
                    # Schedule the actor on this node
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=node_id,
                        soft=False,
                    ),
                    max_concurrency=1000_000_000,  # Prevents the actor from being stuck
                    num_cpus=0,
                    enable_task_events=False,
                ).remote(actor_id)

        await self.scheduling_actors[node_id].ready.remote()  # type: ignore
        return self.scheduling_actors[node_id]

    def preprocessing_callbacks(self) -> dict[str, Callable]:
        """
        Return the preprocessing callbacks for each array.
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
        array = self.arrays[array_name]

        for position, size in chunks:
            array.set_chunk_owner(nb_chunks_per_dim, dtype, position, size, scheduling_actor_id)

    async def chunks_ready(
        self, scheduling_actor_id: int, array_name: str, timestep: Timestep, all_chunks_ref: list[ray.ObjectRef]
    ) -> None:
        """
        Called by the scheduling actors to inform the head actor that the chunks are ready.
        The chunks are not sent.

        Args:
            chunks: Information about the chunks that are ready.
            source_actor: Handle to the scheduling actor owning the chunks.
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
                    array.chunk_refs[timestep] = {}

                    self.new_array_created.set()
                    self.new_array_created.clear()

        is_ready = array.add_chunk_ref(scheduling_actor_id, all_chunks_ref[0], timestep)

        if is_ready:
            self.arrays_ready.put_nowait(
                (
                    array_name,
                    timestep,
                    array.get_full_array(timestep),
                )
            )
            array.fully_defined.set()
            del array.chunk_refs[timestep]

    async def get_next_array(self) -> tuple[str, Timestep, da.Array]:
        array = await self.arrays_ready.get()
        self.new_pending_array_semaphore.release()
        return array

    async def get_preparation_array(self, array_name: str, timestep: Timestep) -> da.Array:
        """
        Return the full Dask array for a given timestep, used for preparation.

        Args:
            array_name: The name of the array.
            timestep: The timestep for which the full array should be returned.
        """
        await self.arrays[array_name].fully_defined.wait()
        return self.arrays[array_name].get_full_array(timestep, is_preparation=True)
