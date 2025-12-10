import random
import time
from collections import Counter
from typing import Callable, Any
import ray
from dask.core import get_dependencies
from deisa.ray.scheduling_actor import ChunkRef, ScheduledByOtherActor
from deisa.ray.types import DoubleRef

type ActorID = str
type GraphKey = Any

def random_partitioning(dsk, scheduling_actors: dict) -> dict[str, int]:
    """
    Partition a Dask task graph randomly across scheduling actors.

    This partitioning strategy assigns tasks randomly to scheduling actors,
    with the exception of ChunkRef tasks which are assigned to their
    designated actor.

    Parameters
    ----------
    dsk : dict
        The Dask task graph dictionary. Keys are task identifiers, values
        are tasks or ChunkRef objects.
    scheduling_actors : dict
        Dictionary mapping actor IDs to their actor handles. The keys are
        used to assign tasks to actors.

    Returns
    -------
    dict[str, int]
        Dictionary mapping task keys to actor IDs. Each task is assigned
        to one scheduling actor.

    Notes
    -----
    The partitioning process:

    1. Counts non-ChunkRef tasks
    2. Creates a list of actor IDs, cycling through actors to distribute
       tasks evenly
    3. Shuffles the actor assignments randomly
    4. Assigns ChunkRef tasks to their designated actor (from ChunkRef.actor_id)
    5. Assigns other tasks to randomly shuffled actors

    This strategy provides load balancing but does not consider task
    dependencies or data locality.
    """
    nb_tasks = len({k for k, v in dsk.items() if not isinstance(v, ChunkRef)})
    nb_scheduling_actors = len(scheduling_actors)
    actor_names = list(scheduling_actors.keys())

    actors = [actor_names[i % nb_scheduling_actors] for i in range(nb_tasks)]
    random.shuffle(actors)

    partition = {}

    for key, val in dsk.items():
        if isinstance(val, ChunkRef):
            partition[key] = val.actor_id
        else:
            partition[key] = actors.pop()

    return partition


def greedy_partitioning(dsk, scheduling_actors: dict) -> dict[str, int]:
    """
    Partition a Dask task graph greedily based on dependencies.

    This partitioning strategy assigns tasks to scheduling actors based on
    their dependencies. Tasks are assigned to the same actor as their
    dependencies when possible, reducing cross-actor communication.

    Parameters
    ----------
    dsk : dict
        The Dask task graph dictionary. Keys are task identifiers, values
        are tasks or ChunkRef objects.
    scheduling_actors : dict
        Dictionary mapping actor IDs to their actor handles. The keys are
        used to assign tasks to actors.

    Returns
    -------
    dict[str, int]
        Dictionary mapping task keys to actor IDs. Each task is assigned
        to one scheduling actor.

    Notes
    -----
    The partitioning process uses a greedy recursive algorithm:

    1. ChunkRef tasks are assigned to their designated actor (from ChunkRef.actor_id)
    2. For other tasks, the algorithm explores dependencies recursively
    3. Tasks are assigned to the actor that appears most frequently among
       their dependencies (using Counter.most_common)
    4. Leaf tasks (no dependencies) are assigned to a random actor

    This strategy aims to minimize cross-actor communication by co-locating
    dependent tasks on the same actor. It provides better data locality than
    random partitioning but may not always produce optimal assignments.

    Examples
    --------
    If task B depends on task A, and task A is assigned to actor 0, then
    task B will also be assigned to actor 0 (assuming no other dependencies
    suggest a different actor).
    """
    partition = {k: -1 for k in dsk.keys()}
    actor_names = list(scheduling_actors.keys())

    def explore(k) -> int:
        if partition[k] != -1:
            return partition[k]

        val = dsk[k]

        if isinstance(val, ChunkRef):
            partition[k] = val.actor_id
        else:
            actors_dependencies = [explore(dep)
                                   for dep in get_dependencies(dsk, k)]

            if not actors_dependencies:
                # The task is a leaf, we use a random actor
                partition[k] = random.choice(actor_names)
            else:
                partition[k] = Counter(
                    actors_dependencies).most_common(1)[0][0]

        return partition[k]

    for key in dsk.keys():
        explore(key)

    return partition

def log(message: str, debug_logs_path: str | None) -> None:
    if debug_logs_path is not None:
        with open(debug_logs_path, "a") as f:
            f.write(f"{time.time()} {message}\n")

def process_keys(keys_needed: list)-> list:
    assert isinstance(keys_needed, list)

    # unnest keys in case of non-aggregate operation
    # TODO: keys are generally a list of needed keys. In case of a simple aggregation, this is a list of one element.
    # however, when doing a non aggregating operation, the keys are wrapped in another list. For example, [[k0, k1], [k2,k3]]
    # investigate whether this is associated to the number of nodes/actors or the chunking?
    def unnest(keys: list)->list:
        if len(keys) == 1:
            if isinstance(keys[0], tuple):
                return [keys[0]]
            else:
                return unnest(keys[0])
        else:
            if isinstance(keys, list):
                res: list = []
                for i in keys:
                    if isinstance(i, list):
                        for j in i:
                            res.append(j)
                    else:
                        res.append(i)
                return res

    return unnest(keys_needed)

def get_scheduling_actors_mapping():
    head_node = ray.get_actor("simulation_head", namespace="deisa_ray")  # noqa: F841

    # Find the scheduling actors
    scheduling_actor_id_to_handle: dict[ActorID, ray.actor.ActorHandle] = ray.get(head_node.list_scheduling_actors.remote())
    assert isinstance(scheduling_actor_id_to_handle, dict)

    return scheduling_actor_id_to_handle

def partition_and_schedule_graph(*, 
                  full_dask_graph: dict, 
                  graph_key_to_actor_id_map: dict[GraphKey, ActorID], 
                  scheduling_actor_id_to_handle: dict[ActorID, ray.actor.ActorHandle],
                  graph_id: int, 
                  ):

    partitioned_graphs: dict[ActorID, dict[GraphKey, Any]] = {
        actor_id: {} for actor_id in scheduling_actor_id_to_handle}

    for k, v in full_dask_graph.items():
        actor_id = graph_key_to_actor_id_map[k]

        partitioned_graphs[actor_id][k] = v

        for dep in get_dependencies(full_dask_graph, k):
            if graph_key_to_actor_id_map[dep] != actor_id:
                partitioned_graphs[actor_id][dep] = ScheduledByOtherActor(
                    graph_key_to_actor_id_map[dep])

    for actorID, actor_handle in scheduling_actor_id_to_handle.items():
        if partitioned_graphs[actorID]:
            # give subgraph to actor for scheduling
            actor_handle.schedule_graph.remote(graph_id, partitioned_graphs[actorID])

def deisa_ray_get(full_dask_graph: dict, keys_needed: list, **kwargs):
    """
    Custom Dask scheduler that partitions and executes graphs on Ray actors.

    This is the main scheduler function that replaces Dask's default scheduler.
    It partitions the task graph across multiple scheduling actors and
    coordinates their execution to compute the requested keys.

    Parameters
    ----------
    dsk : dict
        The Dask task graph dictionary. Keys are task identifiers, values
        are tasks, ChunkRef objects, or other graph nodes.
    keys : list
        List of keys to compute from the task graph. Currently only supports
        a single key (may be nested in a list).
    **kwargs
        Additional keyword arguments. Supported options:

        * ``deisa_ray_partitioning_strategy`` (str, optional): Partitioning
          strategy to use. Options are "random" or "greedy". Default is "greedy".
        * ``deisa_ray_debug_logs`` (str or None, optional): Path to a file
          for debug logging. If None, no logging is performed. Default is None.
        * ``ray_persist`` (bool, optional): If True, returns Ray ObjectRefs
          instead of computed values. Default is False.

    Returns
    -------
    list or list[list]
        If `ray_persist` is False, returns a list containing the computed
        results. The structure matches the input `keys` structure (may be
        nested if keys[0] is a list).
        If `ray_persist` is True, returns a list of Ray ObjectRefs in the
        same structure.

    Raises
    ------
    AssertionError
        If `keys` does not contain exactly one key, or if the head node
        cannot be found, or if scheduling actors are not in the expected
        format.

    Notes
    -----
    The scheduler performs the following steps:

    1. Sorts the graph by keys for deterministic scheduling
    2. Retrieves the head node and scheduling actors
    3. Partitions the graph using the selected strategy (random or greedy)
    4. Creates partitioned graphs for each actor, replacing cross-actor
       dependencies with ScheduledByOtherActor placeholders
    5. Schedules the partitioned graphs on their respective actors
    6. Retrieves the result for the requested key from the appropriate actor
    7. Optionally returns ObjectRefs or computed values

    The function supports debug logging to a file if `deisa_ray_debug_logs`
    is provided. Log entries include timestamps and major scheduling steps.

    Examples
    --------
    >>> # Use greedy partitioning (default)
    >>> result = deisa_ray_get(dsk, ["result_key"])
    >>>
    >>> # Use random partitioning
    >>> result = deisa_ray_get(
    ...     dsk, ["result_key"],
    ...     deisa_ray_partitioning_strategy="random"
    ... )
    >>>
    >>> # Get ObjectRefs instead of computed values
    >>> refs = deisa_ray_get(
    ...     dsk, ["result_key"],
    ...     ray_persist=True
    ... )
    """
    debug_logs_path: str | None = kwargs.get("deisa_ray_debug_logs", None)

    graph_id = random.randint(0, 2**128 - 1)
    partitioning_strategy: Callable = {"random": random_partitioning, "greedy": greedy_partitioning}[
        kwargs.get("deisa_ray_partitioning_strategy", "greedy")
    ]

    scheduling_actor_id_to_handle: dict[ActorID, ray.actor.ActorHandle] = get_scheduling_actors_mapping()

    full_dask_graph = {k: v for k, v in sorted(full_dask_graph.items())}
    graph_key_to_actor_id_map: dict[GraphKey, ActorID] = partitioning_strategy(full_dask_graph, scheduling_actor_id_to_handle)

    partition_and_schedule_graph(
        full_dask_graph = full_dask_graph, 
        graph_key_to_actor_id_map=graph_key_to_actor_id_map, 
        scheduling_actor_id_to_handle=scheduling_actor_id_to_handle,
        graph_id = graph_id
        )

    keys_needed = process_keys(keys_needed)

    # repack keys in case of non-aggregate operation

    result_refs: list[DoubleRef] = []

    for key in keys_needed:
        actor_id: ActorID = graph_key_to_actor_id_map[key]
        actor_handle = scheduling_actor_id_to_handle[actor_id]
        result_refs.append(actor_handle.get_value.remote(graph_id, key))

    # NOTE : not sure how to handle persist
    if kwargs.get("ray_persist"):
        return result_refs

    res = []
    for double_ref in result_refs:
        res.append(ray.get(ray.get(double_ref)))
    return [res]