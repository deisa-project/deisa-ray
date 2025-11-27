import asyncio
import pickle
import time
import random
from dataclasses import dataclass

import numpy as np
import ray
import ray.actor
import ray.util.dask.scheduler

from deisa.ray import Timestep
from deisa.ray._async_dict import AsyncDict


@dataclass
class ChunkRef:
    """
    Represents a chunk of an array in a Dask task graph.

    The task corresping to this object must be scheduled by the actor who has the actual
    data. This class is used since Dask tends to inline simple tuples. This may change
    in newer versions of Dask.
    """

    actor_id: int
    array_name: str  # The real name, without the timestep
    timestep: Timestep
    position: tuple[int, ...]

    # Set for one chunk only.
    _all_chunks: ray.ObjectRef | None = None


@dataclass
class ScheduledByOtherActor:
    """
    Represents a task that is scheduled by another actor in the part of the task graph sent to an actor.
    """

    actor_id: int


class GraphInfo:
    """
    Information about graphs and their scheduling.
    """

    def __init__(self):
        self.scheduled_event = asyncio.Event()
        self.refs: dict[str, ray.ObjectRef] = {}


@ray.remote(num_cpus=0, enable_task_events=False)
def patched_dask_task_wrapper(func, repack, key, ray_pretask_cbs, ray_posttask_cbs, *args, first_call=True):
    """
    Patched version of the original dask_task_wrapper function.

    This version received ObjectRefs first, and calls itself a second time to unwrap the ObjectRefs.
    The result is an ObjectRef.

    TODO can probably be rewritten without copying the whole function
    """

    if first_call:
        assert all([isinstance(a, ray.ObjectRef) for a in args])
        # Use one CPU for the actual computation
        return patched_dask_task_wrapper.options(num_cpus=1).remote(
            func, repack, key, ray_pretask_cbs, ray_posttask_cbs, *args, first_call=False
        )

    if ray_pretask_cbs is not None:
        pre_states = [cb(key, args) if cb is not None else None for cb in ray_pretask_cbs]
    repacked_args, repacked_deps = repack(args)
    # Recursively execute Dask-inlined tasks.
    actual_args = [ray.util.dask.scheduler._execute_task(a, repacked_deps) for a in repacked_args]
    # Execute the actual underlying Dask task.
    result = func(*actual_args)

    if ray_posttask_cbs is not None:
        for cb, pre_state in zip(ray_posttask_cbs, pre_states):
            if cb is not None:
                cb(key, result, pre_state)

    return result


@ray.remote(num_cpus=0, enable_task_events=False)
def remote_ray_dask_get(dsk, keys):
    import ray.util.dask

    # Monkey-patch Dask-on-Ray
    ray.util.dask.scheduler.dask_task_wrapper = patched_dask_task_wrapper

    return ray.util.dask.ray_dask_get(dsk, keys, ray_persist=True)


class _ArrayTimestep:
    def __init__(self):
        # Triggered when all the chunks are ready
        self.chunks_ready_event: asyncio.Event = asyncio.Event()

        # {position: chunk}
        self.local_chunks: AsyncDict[tuple[int, ...], ray.ObjectRef | bytes] = AsyncDict()


class _Array:
    def __init__(self):
        # Indicates if set_owned_chunks method has been called for this array.
        self.is_registered = False

        # Chunks owned by this actor for this array.
        # {(chunk position, chunk size), ...}
        self.owned_chunks: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

        self.timesteps: AsyncDict[Timestep, _ArrayTimestep] = AsyncDict()


def get_ready_actor_with_retry(name, namespace, deadline_s=180):
    start, delay = time.time(), 0.2
    while True:
        try:
            actor = ray.get_actor(name=name, namespace=namespace)
            # ready gate
            # TODO for even more reliability, in the future we should handle
            # actor exists, but unavailable
            # actor exists, crashed, need to recreate
            ray.get(actor.ready.remote())
            return actor
        except ValueError:
            if time.time() - start > deadline_s:
                raise TimeoutError(f"{namespace}/{name} not found in {deadline_s}s")
            time.sleep(delay + random.random() * 0.1)
            delay = min(delay * 1.5, 5.0)


@ray.remote
class SchedulingActor:
    """
    Actor in charge of gathering ObjectRefs and scheduling the tasks produced by the head node.
    """

    async def __init__(self, actor_id: int) -> None:
        self.actor_id = actor_id
        self.actor_handle = ray.get_runtime_context().current_actor

        self.head = get_ready_actor_with_retry(name="simulation_head", namespace="deisa_ray")
        await self.head.register_scheduling_actor.remote(actor_id, self.actor_handle)
        self.scheduling_actors: list[ray.actor.ActorHandle] = []

        # For collecting chunks
        self.arrays: AsyncDict[str, _Array] = AsyncDict()

        # For scheduling
        self.graph_infos: AsyncDict[int, GraphInfo] = AsyncDict()

    def preprocessing_callbacks(self) -> ray.ObjectRef:
        # return obect ref
        p_clbs = self.head.preprocessing_callbacks.remote()
        assert isinstance(p_clbs, ray.ObjectRef)
        return p_clbs

    def ready(self) -> None:
        pass

    def _pack_object_ref(self, refs: list[ray.ObjectRef]):
        """
        Used to create an ObjectRef containing the given ObjectRef.
        This allows having the expected format in the task graph.

        This is a method instead of a function with `num_cpus=0` to avoid starting many
        new workers.
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
        if array_name not in self.arrays:
            self.arrays[array_name] = _Array()
        array = self.arrays[array_name]

        if timestep not in array.timesteps:
            array.timesteps[timestep] = _ArrayTimestep()
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
        graph_info = await self.graph_infos.wait_for_key(graph_id)

        await graph_info.scheduled_event.wait()
        return await graph_info.refs[key]
