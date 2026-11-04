"""Regression test: ``SchedulingActor.get_value`` must resolve literal (non-task) graph keys.

``ray_dask_get(..., ray_persist=True)`` wraps real TASKS in a ref-of-ref, but returns the PLAIN VALUE for graph keys
that are literals / aliases (Dask ``DataNode`` / ``Alias``) -- for example the array built by ``da.fft.fftfreq`` or any
``from_array`` constant. ``schedule_graph`` used to store that plain value in ``GraphInfo.refs`` verbatim, while
``get_value`` does a single-level ``await ref`` to unwrap the ref-of-ref. As soon as another scheduling actor needed
such a key, the await raised ``TypeError: object numpy.ndarray can't be used in 'await' expression`` (or the resolution
stalled), which is the intermittent failure seen by real callbacks whose graphs mix chunk data with computed constants.

This test drives the two actors directly, so it is deterministic regardless of how the partitioner happens to distribute
literal keys.
"""

from __future__ import annotations

import numpy as np
import ray
from dask._task_spec import DataNode
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from deisa.ray.head_node import HeadNodeActor
from deisa.ray.scheduling_actor import SchedulingActor
from deisa.ray.utils import get_head_actor_options


def test_get_value_resolves_a_literal_key(ray_multinode_cluster) -> None:
    worker_node_ids = [
        node.node_id for node in ray_multinode_cluster["cluster"].list_all_nodes() if not node.is_head()
    ]

    head = HeadNodeActor.options(**get_head_actor_options()).remote()
    ray.get(head.ready.remote())

    owner = SchedulingActor.options(
        name="literal-owner",
        namespace="deisa_ray",
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=worker_node_ids[0], soft=False),
    ).remote("literal-owner")
    ray.get(owner.ready.remote())

    graph_id = 4242
    value = np.arange(4, dtype=np.float64)
    ray.get(owner.schedule_graph.remote(graph_id, {"lit": DataNode("lit", value)}))

    # A literal key has no task, so ray_dask_get returns the raw value; get_value must still return a ref the caller
    # can unwrap exactly as it does for task keys.
    inner = ray.get(owner.get_value.remote(graph_id, "lit"))

    assert isinstance(inner, ray.ObjectRef)
    np.testing.assert_array_equal(ray.get(inner), value)
