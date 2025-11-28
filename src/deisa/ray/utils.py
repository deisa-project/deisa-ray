from typing import Dict
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

def get_system_metadata()-> Dict: 
    return {}

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

