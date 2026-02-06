import asyncio

import dask.array as da
import numpy as np
import ray

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
    def __init__(self, *,n_sim_nodes: int, max_simulation_ahead: int = 1) -> None:
        """
        Initialize synchronization primitives and bookkeeping containers.

        Notes
        -----
        The semaphore that limits in-flight arrays is instantiated in
        :meth:`register_arrays`. Here we only create the shared state that
        persists for the lifetime of the actor (registered array metadata,
        the queue of completed arrays, and the event used to signal array
        creation).
        """
        # For each ID of a actor_handle, the corresponding scheduling actor
        self.needed_arrays = None
        self.scheduling_actors: dict[str, RayActorHandle] = {}
        self.n_sim_nodes_counter = n_sim_nodes

        # TODO: document what this event signals and update documentation
        self.new_array_created: dict[str, asyncio.Event] = {}
        self.max_simulation_ahead = max_simulation_ahead
        self.semaphore_per_array = {}

        # All the newly created arrays
        self.arrays_ready: asyncio.Queue[tuple[str, Timestep, da.Array]] = asyncio.Queue()
        self.arrays_needed_by_analytics: dict[str, DaskArrayData] = {}
        self.analytics_ready_for_execution: asyncio.Event = asyncio.Event()

    def set_analytics_ready_for_execution(self):
        self.analytics_ready_for_execution.set()
        # self.analytics_ready_for_execution.clear()

    async def wait_until_analytics_ready(self):
        await self.analytics_ready_for_execution.wait()

    # TODO rename or move creation of global container elsewhere
    def register_array_needed_by_analytics(self, array_names: set[str]) -> None:
        """
        Register array definitions and set back-pressure on pending timesteps.

        Parameters
        ----------
        arrays_definitions : list[tuple[str, Callable]]
            Sequence of ``(name)`` pairs for each array
            produced by the simulation. Each entry becomes a
            :class:`~deisa.ray.types.DaskArrayData` instance.
        max_pending_arrays : int, optional
            Upper bound on the number of array timesteps that may be created
            but not yet consumed. Used to throttle simulations that outrun
            analytics. Default is ``1_000_000_000``.

        Notes
        -----
        This method must be called before any scheduling actors register
        themselves or send chunks. It creates a semaphore that gates the
        creation of new timesteps and populates ``registered_arrays`` with
        the provided definitions.
        """
        # regulate how far ahead sim can go wrt to analytics
        for name in array_names:
            self.arrays_needed_by_analytics[name] = DaskArrayData(name)
            self.semaphore_per_array[name] = asyncio.Semaphore(self.max_simulation_ahead + 1)
            self.new_array_created[name] = asyncio.Event()

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
            if self.n_sim_nodes_counter>0:
                self.scheduling_actors[actor_id] = actor_handle
                self.n_sim_nodes_counter-=1
                return True
            else:
                return False

    def get_n_sim_nodes_counter(self):
        return self.n_sim_nodes_counter

    def register_partial_array(
        self,
        actor_id_who_owns: int,
        array_name: str,
        dtype: np.dtype,
        nb_chunks_per_dim: tuple[int, ...],
        chunks_meta: list[tuple[int, tuple[int, ...], tuple[int, ...]]],  # [(chunk position, chunk size), ...]
    ):
        """
        Register chunk ownership for a single array on behalf of one actor.

        Parameters
        ----------
        actor_id_who_owns : int
            Scheduling actor ID that owns the provided chunks.
        array_name : str
            Name of the array being registered.
        dtype : np.dtype
            NumPy dtype for all chunks owned by the actor.
        nb_chunks_per_dim : tuple[int, ...]
            Global chunk grid shape (number of chunks per dimension).
        chunks_meta : list[tuple[int, tuple[int, ...], tuple[int, ...]]]
            Iterable of ``(bridge_id, chunk_position, chunk_size)`` tuples
            describing each owned chunk.

        Notes
        -----
        Called once per scheduling actor per array to let the head actor know
        who owns each chunk. The metadata is forwarded to
        :class:`~deisa.ray.types.DaskArrayData` and later used to build the
        Dask arrays when chunk payloads arrive.
        """
        # TODO no checks done for when all nodeactors have called this
        # TODO missing check that analytics and sim have required/set same name of arrays, otherwise array is created and nothing happens
        array = self.arrays_needed_by_analytics[array_name]

        for bridge_id, position, size in chunks_meta:
            array.update_meta(nb_chunks_per_dim, dtype, position, size, actor_id_who_owns, bridge_id)

    def exchange_config(self, config: dict) -> None:
        """
        Store runtime configuration flags for the head actor.

        Parameters
        ----------
        config : dict
            Configuration dictionary expected to contain the key
            ``\"experimental_distributed_scheduling_enabled\"``. Additional
            keys are stored unchanged for downstream consumers.

        Notes
        -----
        This method is invoked by the window handler during setup to keep
        the head actor aware of cluster-wide feature flags.
        """
        self.config = config
        self._experimental_distributed_scheduling_enabled = config["experimental_distributed_scheduling_enabled"]

    async def chunks_ready(
        self, array_name: str, timestep: Timestep, pos_to_ref: dict[tuple, ray.ObjectRef], actor_id: str
    ) -> None:
        """
        Receive chunk references for a timestep and enqueue the full array when complete.

        Scheduling actors call this once they have prepared their owned
        chunks for ``timestep``. The head actor waits until the timestep is
        registered, records the chunk references, and when all owners have
        reported marks the array as ready for analytics consumption.

        Parameters
        ----------
        array_name : str
            The name of the array these chunks belong to.
        timestep : Timestep
            The timestep these chunks belong to.
        pos_to_ref : dict[tuple, ray.ObjectRef]
            Mapping from chunk position to Ray ObjectRef (double refs) for
            the chunk payloads owned by the calling actor.

        Notes
        -----
        If the timestep entry is missing, the method waits either for the
        pending-array semaphore (creating the entry itself) or for another
        actor to create it. When :class:`~deisa.ray.types.DaskArrayData`
        reports the timestep as complete, the assembled Dask array (or
        distributed-scheduling graph) is put into ``arrays_ready``. The
        semaphore is released when analytics later consume the array.
        """
        array = self.arrays_needed_by_analytics[array_name]
        semaphore = self.semaphore_per_array[array_name]
        created_event = self.new_array_created[array_name]

        # Ensure the timestep entry exists
        while timestep not in array.chunk_refs:
            try_to_create_timestep_task = asyncio.create_task(semaphore.acquire())
            wait_for_another_to_create_task = asyncio.create_task(created_event.wait())

            done, pending = await asyncio.wait(
                {try_to_create_timestep_task, wait_for_another_to_create_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()

            if try_to_create_timestep_task in done:
                if timestep in array.chunk_refs:
                    # NOTE : cannot happen since that in the stalling case,
                    # one of the scheduling actor dont add his chunks
                    # -> is_ready is always false
                    # -> nothing is put in the queue, get_next_array() never executed
                    # -> semaphore never released
                    # -> so try_to_create_timestep_task cant finish and wait_for_another_to_create_task keep waiting -> stall forever
                    #
                    # Another actor created the entry while we were waiting
                    semaphore.release()
                else:
                    array.chunk_refs[timestep] = []
                    created_event.set()
                    # NOTE : If the first scheduling actor that create the timestep
                    # clear the event before one of the others scheduling actors create
                    # the asyncio task that wait on it it will wait forever
                    #
                    created_event.clear()
                    #
                # NOTE : possible solution :
                # - release semaphore at different place
                # - clear the event later ?

        # Collect chunk refs and store them in Ray
        chunks = list(pos_to_ref.values())
        ref_to_list_of_chunks = ray.put(chunks)

        # ray.get(ref_to_list_of_chunks) -> [ref_of_ref_chunk_i, ref_ref_chunk_i+1, ...] (belonging to actor that owns it)
        # so I unpack it and give this ray ref.
        is_ready = array.add_chunk_ref(ref_to_list_of_chunks, timestep, pos_to_ref)

        if is_ready:
            self.arrays_ready.put_nowait(
                (
                    array_name,
                    timestep,
                    array.get_full_array(
                        timestep, distributing_scheduling_enabled=self._experimental_distributed_scheduling_enabled
                    ),
                )
            )
            # TODO for now, only used when doing distributed scheduling, but in theory could be
            # used with centralized scheduling too
            if self._experimental_distributed_scheduling_enabled:
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
        Get and return the next ready array from the queue.

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
        queue. When an array is retrieved, the pending-array semaphore is
        released so new timesteps may be created. Called by analytics
        components to pull work in order.
        """
        array = await self.arrays_ready.get()
        self.semaphore_per_array[array[0]].release()
        return array
