import random
import warnings
from collections import Counter
from typing import Callable

import ray
import dask
from dask.core import get_dependencies
try:
    from dask._task_spec import Alias, DataNode, Task, TaskRef, Mapping
except ImportError:
    warnings.warn(
        "Dask on Ray is available only on dask>=2024.11.0, "
        f"you are on version {dask.__version__}."
    )

from doreisa._scheduling_actor import ChunkRef, ScheduledByOtherActor


def explore_task_args(task: Task) -> int:
    for arg in task.args:
        if isinstance(arg, dict):
            for value in arg.values():
                if isinstance(value, DataNode) and isinstance(value.value, ChunkRef):
                    return value.value.actor_id
    return -2


# TODO : partition assignation to actors may be better
def random_partitioning(dsk, nb_scheduling_actors: int) -> dict[str, int]:
    nb_tasks = len({k for k, v in dsk.items() if isinstance(v, Task)})
    actors = [i % nb_scheduling_actors for i in range(nb_tasks)]
    partition = {}

    for key, val in dsk.items():
        if isinstance(val, DataNode) and isinstance(val.value, ChunkRef):
            partition[key] = val.value.actor_id
        elif isinstance(val, Task):
            actors_id = explore_task_args(val)
            if actors_id != -2:
                partition[key] = actors_id
            else:
                partition[key] = actors[random.randint(0, len(actors)-1)]
        else:
            partition[key] = actors[random.randint(0, len(actors)-1)]
    return partition


def greedy_partitioning(dsk, nb_scheduling_actors: int) -> dict[str, int]:
    partition = {k: -1 for k in dsk.keys()}

    def explore(k) -> int:
        if partition[k] != -1:
            return partition[k]

        val = dsk[k]

        if isinstance(val, DataNode) and isinstance(val.value, ChunkRef):
            partition[k] = val.value.actor_id
        elif isinstance(val, Task) and explore_task_args(val) != -2:
            actors_id = explore_task_args(val)
            if actors_id != -2:
                partition[k] = actors_id
        else:
            actors_dependencies = [explore(dep)
                                   for dep in get_dependencies(dsk, k)]
            if not actors_dependencies:
                # The task is a leaf, we use a random actor
                partition[k] = random.randint(0, nb_scheduling_actors - 1)
            else:
                partition[k] = Counter(
                    actors_dependencies).most_common(1)[0][0]

        return partition[k]

    for key in dsk.keys():
        explore(key)

    return partition


def doreisa_get(dsk, keys, **kwargs):
    # debug_logs_path: str | None = kwargs.get("doreisa_debug_logs", None)
    debug_logs_path = "./logs"

    def log(message: str, debug_logs_path: str | None) -> None:
        if debug_logs_path is not None:
            with open(debug_logs_path, "a") as f:
                f.write(f"{message}\n")

    partitioning_strategy: Callable = {"random": random_partitioning, "greedy": greedy_partitioning}[
        kwargs.get("doreisa_partitioning_strategy", "greedy")
    ]

    log("1. Begin Doreisa scheduler", debug_logs_path)

    dsk = dsk.__dask_graph__()

    log(f"\n[DEBUG] Initial graph = {dsk}\n", debug_logs_path)

    head_node = ray.get_actor("simulation_head", namespace="doreisa")  # noqa: F841

    # TODO this will not work all the time
    assert isinstance(keys, list) and len(keys) == 1
    if isinstance(keys[0], list):
        assert len(keys[0]) == 1
        key = keys[0][0]
    else:
        key = keys[0]

    # Find the scheduling actors
    scheduling_actors = ray.get(head_node.list_scheduling_actors.remote())

    partition = partitioning_strategy(dsk, len(scheduling_actors))

    log("2. Graph partitioning done", debug_logs_path)

    partitioned_graphs: dict[int, dict] = {
        actor_id: {} for actor_id in range(len(scheduling_actors))}

    for k, v in dsk.items():
        actor_id = partition[k]
        partitioned_graphs[actor_id][k] = v

        for dep in get_dependencies(dsk, k):
            if partition[dep] != actor_id:
                partitioned_graphs[actor_id][dep] = ScheduledByOtherActor(
                    partition[dep])

    log("3. Partitioned graphs created", debug_logs_path)
    log(f"[DEBUG] Partitioned graph = {partitioned_graphs}", debug_logs_path)

    graph_id = random.randint(0, 2**128 - 1)

    for id, actor in enumerate(scheduling_actors):
        if partitioned_graphs[id]:
            actor.schedule_graph.remote(graph_id, partitioned_graphs[id])

    log("4. Graph scheduled", debug_logs_path)

    res_ref = scheduling_actors[partition[key]].get_value.remote(graph_id, key)

    if kwargs.get("ray_persist"):
        log(f"[DEBUG, doreisa_get] keys[0] = {keys[0]}", "./logs")
        if isinstance(keys[0], list):
            # TODO : verify that the return value works
            # if the condition is verified
            return [[res_ref]]
        log(f"[DEBUG, doreisa_get] res_ref = {res_ref}", "./logs")
        log(f"[DEBUG, doreisa_get] ray.get(res_ref) = {
            ray.get(res_ref)}", "./logs")
        log(f"[DEBUG, doreisa_get] ray.get(ray.get(res_ref) = {
            ray.get(ray.get(res_ref))}", "./logs")
        return [ray.get(ray.get(res_ref))]

    res = ray.get(ray.get(res_ref))

    log("5. End Doreisa scheduler", debug_logs_path)

    if isinstance(keys[0], list):
        return [[res]]
    return [res]
