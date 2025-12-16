import asyncio
from typing import Callable

import dask.array as da
import numpy as np
import ray
import ray.actor

from deisa.ray import Timestep
from deisa.ray.types import DaskArrayData, RayActorHandle


@ray.remote
class HeadNodeActor:
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
    scheduling_actors : dict[str, RayActorHandle]
        Mapping from scheduling actor IDs to their actor handles.
    new_pending_array_semaphore : asyncio.Semaphore
        Semaphore used to limit the number of pending arrays.
    new_array_created : asyncio.Event
        Event set when a new array timestep is created.
    arrays : dict[str, DaskArrayData]
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

    # TODO: Discuss if max_pending_arrays should be here or in register callback. In that case, what
    # should happen when the freqs are diff and max_pending_arrays are diff too? When does the sim
    # stop?''
    def __init__(self) -> None:
        # For each ID of a actor_handle, the corresponding scheduling actor
        self.scheduling_actors: dict[str, RayActorHandle] = {}

        # TODO: document what this event signals and update documentation
        self.new_array_created = asyncio.Event()

        # All the newly created arrays
        self.arrays_ready: asyncio.Queue[tuple[str, Timestep, da.Array]] = asyncio.Queue()
        self.registered_arrays: dict[str, DaskArrayData] = {}

    # TODO rename or move creation of global container elsewhere
    def register_arrays(
        self, arrays_definitions: list[tuple[str, Callable]], max_pending_arrays: int = 1_000_000_000
    ) -> None:
        """
        Register array definitions and initialize bookkeeping structures.

        Parameters
        ----------
        arrays_definitions : list[tuple[str, Callable]]
            Sequence of (name, preprocessing_callback) pairs for each array
            produced by the simulation.
        max_pending_arrays : int, optional
            Upper bound on arrays that may be in-flight (built or waiting
            to be collected). Acts as back-pressure for the simulation.
            Default is ``1_000_000_000``.
        """
        # regulate how far ahead sim can go wrt to analytics
        self.new_pending_array_semaphore = asyncio.Semaphore(max_pending_arrays)

        for name, f_preprocessing in arrays_definitions:
            self.registered_arrays[name] = DaskArrayData(name, f_preprocessing)

    def list_scheduling_actors(self) -> dict[str, RayActorHandle]:
        """
        Return the list of scheduling actors.

        Returns
        -------
        Dict[RayActorHandle]
            Dictionary of actor_id to actor handles for all registered scheduling actors.
        """
        return self.scheduling_actors

    async def register_scheduling_actor(self, actor_id: str, actor_handle: RayActorHandle):
        """
        Register a scheduling actor that has been created.

        Parameters
        ----------
        actor_id : str
            Unique identifier for the scheduling actor.
        actor_handle : RayActorHandle
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
        return {name: array.f_preprocessing for name, array in self.registered_arrays.items()}

    def register_partial_array(
        self,
        actor_id_who_owns: int,
        array_name: str,
        dtype: np.dtype,
        nb_chunks_per_dim: tuple[int, ...],
        chunks_meta: list[tuple[int, tuple[int, ...], tuple[int, ...]]],  # [(chunk position, chunk size), ...]
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
        chunks : list[int, tuple[tuple[int, ...], tuple[int, ...]]]
            List of tuples, each containing (chunk_position, chunk_size).
            The chunk_position is a tuple of indices, and chunk_size is a
            tuple of sizes along each dimension.

        Notes
        -----
        This method is called by scheduling actors to inform the head actor
        which chunks they are responsible for. This information is used to
        track array construction progress.
        """
        # TODO no checks done for when all nodeactors have called this
        # TODO missing check that analytics and sim have required/set same name of arrays, otherwise array is created and nothing happens
        array = self.registered_arrays[array_name]

        for bridge_id, position, size in chunks_meta:
            array.update_meta(nb_chunks_per_dim, dtype, position, size, actor_id_who_owns, bridge_id)

    async def chunks_ready(self, array_name: str, timestep: Timestep, pos_to_ref: dict[tuple, ray.ObjectRef]) -> None:
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
        array = self.registered_arrays[array_name]

        # long story short, this creates an empty array.chunk_refs[timestep] = []
        while timestep not in array.chunk_refs:
            # TODO what happens if new_array_created for an array of a prev iter is set?
            # could it influence this iter?
            # need to reason more about this condition
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
        chunks = [val for val in pos_to_ref.values()]
        ref_to_list_of_chunks = ray.put(chunks)

        # ray.get(ref_to_list_of_chunks) -> [ref_of_ref_chunk_i, ref_ref_chunk_i+1, ...] (belonging to actor that owns it)
        # so I unpack it and give this ray ref.
        is_ready = array.add_chunk_ref(ref_to_list_of_chunks, timestep, pos_to_ref)

        distributed_scheduling_enabled = True
        if is_ready:
            self.arrays_ready.put_nowait(
                (
                    array_name,
                    timestep,
                    array.get_full_array(timestep, distributing_scheduling_enabled=distributed_scheduling_enabled),
                )
            )
        # TODO Just used for preparation stuff
        if distributed_scheduling_enabled:
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

    # TODO there is a single queue for all arrays.
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
        await self.registered_arrays[array_name].fully_defined.wait()
        return self.registered_arrays[array_name].get_full_array(timestep, is_preparation=True)
