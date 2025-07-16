import random
import time
from collections import Counter
from typing import Callable

import ray
import ray.actor
from dask.core import get_dependencies

from doreisa._scheduling_actor import ChunkRef, ScheduledByOtherActor


def random_partitioning(dsk, nb_scheduling_actors: int) -> dict[str, int]:
    nb_tasks = len({k for k, v in dsk.items() if not isinstance(v, ChunkRef)})

    actors = [i % nb_scheduling_actors for i in range(nb_tasks)]
    random.shuffle(actors)

    partition = {}

    for key, val in dsk.items():
        if isinstance(val, ChunkRef):
            partition[key] = val.actor_id
        else:
            partition[key] = actors.pop()

    return partition


def greedy_partitioning(dsk, nb_scheduling_actors: int) -> dict[str, int]:
    partition = {k: -1 for k in dsk.keys()}

    def explore(k) -> int:
        if partition[k] != -1:
            return partition[k]

        val = dsk[k]

        if isinstance(val, ChunkRef):
            partition[k] = val.actor_id
        else:
            actors_dependencies = [explore(dep) for dep in get_dependencies(dsk, k)]

            if not actors_dependencies:
                # The task is a leaf, we use a random actor
                partition[k] = random.randint(0, nb_scheduling_actors - 1)
            else:
                partition[k] = Counter(actors_dependencies).most_common(1)[0][0]

        return partition[k]

    for key in dsk.keys():
        explore(key)

    return partition


def doreisa_get(dsk, keys, **kwargs):
    debug_logs_path: str | None = kwargs.get("doreisa_debug_logs", None)

    def log(message: str, debug_logs_path: str | None) -> None:
        if debug_logs_path is not None:
            with open(debug_logs_path, "a") as f:
                f.write(f"{time.time()} {message}\n")

    partitioning_strategy: Callable = {"random": random_partitioning, "greedy": greedy_partitioning}[
        kwargs.get("doreisa_partitioning_strategy", "greedy")
    ]

    log("1. Begin Doreisa scheduler", debug_logs_path)

    # Sort the graph by keys to make scheduling deterministic
    dsk = {k: v for k, v in sorted(dsk.items())}

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

    partitioned_graphs: dict[int, dict] = {actor_id: {} for actor_id in range(len(scheduling_actors))}

    for k, v in dsk.items():
        actor_id = partition[k]

        partitioned_graphs[actor_id][k] = v

        for dep in get_dependencies(dsk, k):
            if partition[dep] != actor_id:
                partitioned_graphs[actor_id][dep] = ScheduledByOtherActor(partition[dep])

    log("3. Partitioned graphs created", debug_logs_path)

    graph_id = random.randint(0, 2**128 - 1)

    for id, actor in enumerate(scheduling_actors):
        if partitioned_graphs[id]:
            actor.schedule_graph.remote(graph_id, partitioned_graphs[id])

    log("4. Graph scheduled", debug_logs_path)

    res_ref = scheduling_actors[partition[key]].get_value.remote(graph_id, key)

    clear_graph.remote(
        [actor for id, actor in enumerate(scheduling_actors) if partitionned_graphs[id]], res_ref, graph_id
    )

    if kwargs.get("ray_persist"):
        if isinstance(keys[0], list):
            return [[res_ref]]
        return [res_ref]

    res = ray.get(ray.get(res_ref))

    log("5. End Doreisa scheduler", debug_logs_path)

    if isinstance(keys[0], list):
        return [[res]]
    return [res]


@ray.remote(max_retries=0, num_cpus=0)
def clear_graph(scheduling_actors: list[ray.actor.ActorHandle], res: ray.ObjectRef, graph_id: int) -> None:
    # Wait until the result is ready
    ray.wait([res], fetch_local=False)

    # Clear the graph
    ray.get([actor.clear_graph.remote(graph_id) for actor in scheduling_actors])
