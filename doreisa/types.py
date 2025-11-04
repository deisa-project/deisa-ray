import asyncio
from dataclasses import dataclass

import ray
import ray.util.dask.scheduler

from doreisa import Timestep

@dataclass
class ChunkRef:
    """
    Represents a chunk of an array in a Dask task graph.

    The task corresponding to this object must be scheduled by the actor who has the actual
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
    Represents a task that is scheduled by another actor in the part of the task graph sent to an 
    actor.
    """

    actor_id: int


class GraphInfo:
    """
    Information about graphs and their scheduling.
    """

    def __init__(self):
        self.scheduled_event = asyncio.Event()
        self.refs: dict[str, ray.ObjectRef] = {}
