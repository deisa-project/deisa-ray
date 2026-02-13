import random
import time
from typing import Dict, Any
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


def get_node_actor_options(name: str, namespace: str) -> Dict[str, Any]:
    """Return Ray options used to create (or get) a node scheduling actor.

    Parameters
    ----------
    name : str
        Actor name to use for the node actor.
    namespace : str
        Ray namespace where the actor will live.

    Returns
    -------
    dict
        Dictionary of options to be passed to ``SchedulingActor.options``.

    Notes
    -----
    The options use ``get_if_exists=True`` to avoid race conditions when
    several bridges on the same node attempt to create the same actor.
    The actor is configured with:

    - ``lifetime='detached'`` so it survives the creating task
    - ``num_cpus=0`` so it does not reserve CPU resources
    - a very large ``max_concurrency`` because the actor is async-only
      and used mainly as a coordination point.
    """
    return {
        "name": name,
        "namespace": namespace,
        "lifetime": "detached",
        "get_if_exists": True,
        # WARNING: if not using async actor this will make OS try to spawn many threads
        # and blow everything up. Scheduling actors need to be async because of this.
        "max_concurrency": 1_000_000_000,
        "num_cpus": 0,
        "enable_task_events": False,
    }


def get_system_metadata() -> Dict:
    """
    Return system-level metadata placeholder.

    Notes
    -----
    Currently returns an empty dictionary; the hook exists to keep backward
    compatibility with callers expecting environment metadata.
    """
    return {}


async def get_ready_actor_with_retry(name, namespace, deadline_s=180):
    """
    Get a Ray actor by name with retry logic and readiness check.

    This function attempts to retrieve a Ray actor by name and namespace,
    checking that it is ready before returning. It implements exponential
    backoff retry logic with a deadline.

    Parameters
    ----------
    name : str
        The name of the actor to retrieve.
    namespace : str
        The namespace of the actor.
    deadline_s : float, optional
        Maximum time in seconds to wait for the actor to become available.
        Default is 180.

    Returns
    -------
    RayActorHandle
        The handle to the ready actor.

    Raises
    ------
    TimeoutError
        If the actor is not found or not ready within the deadline.

    Notes
    -----
    The function uses exponential backoff with jitter for retries. The delay
    starts at 0.2 seconds and increases by a factor of 1.5 up to a maximum
    of 5.0 seconds. A small random jitter (0-0.1 seconds) is added to avoid
    thundering herd problems.
    """
    start, delay = time.time(), 0.2
    while True:
        try:
            actor = ray.get_actor(name=name, namespace=namespace)
            # ready gate
            # TODO for even more reliability, in the future we should handle
            # actor exists, but unavailable
            # actor exists, crashed, need to recreate
            await actor.ready.remote()
            return actor
        except ValueError:
            if time.time() - start > deadline_s:
                raise TimeoutError(f"{namespace}/{name} not found in {deadline_s}s")
            time.sleep(delay + random.random() * 0.1)
            delay = min(delay * 1.5, 5.0)


def get_head_node_id() -> str:
    """
    Get the node ID of the Ray cluster head node.

    Returns
    -------
    str
        The node ID of the head node.

    Raises
    ------
    AssertionError
        If there is not exactly one head node in the cluster.

    Notes
    -----
    This function queries Ray's state API to find the head node. It assumes
    there is exactly one head node in the cluster.
    """
    from ray.util import state

    nodes = state.list_nodes(filters=[("is_head_node", "=", True)])

    assert len(nodes) == 1, "There should be exactly one head node"

    return nodes[0].node_id


def get_head_actor_options() -> dict:
    """
    Return the options that should be used to start the head actor.

    Returns
    -------
    dict
        Dictionary of Ray actor options including:
        - name: "simulation_head"
        - namespace: "deisa_ray"
        - scheduling_strategy: NodeAffinitySchedulingStrategy for the head node
        - max_concurrency: Very high value to prevent blocking
        - lifetime: "detached" to persist beyond function scope
        - enable_task_events: False for performance

    Notes
    -----
    The head actor is scheduled on the head node with a detached lifetime
    to ensure it persists. High concurrency is set to prevent the actor
    from being blocked when gathering many references.
    """
    return dict(
        # The workers will be able to access to this actor using its name
        name="simulation_head",
        namespace="deisa_ray",
        # Schedule the actor on this node
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=get_head_node_id(),
            soft=False,
        ),
        # Prevents the actor from being stuck when it needs to gather many refs
        max_concurrency=1000_000_000,
        # Prevents the actor from being deleted when the function ends
        lifetime="detached",
        # Disabled for performance reasons
        enable_task_events=False,
    )


def log(message: str, debug_logs_path: str | None) -> None:
    """
    Append a timestamped debug message to ``debug_logs_path`` if provided.

    Parameters
    ----------
    message : str
        Text to append.
    debug_logs_path : str or None
        Destination file path. If ``None`` logging is skipped.
    """
    if debug_logs_path is not None:
        with open(debug_logs_path, "a") as f:
            f.write(f"{time.time()} {message}\n")
