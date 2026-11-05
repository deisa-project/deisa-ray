"""
Regression tests for the distributed-scheduling graph partitioners.

The partitioners in :mod:`deisa.ray._scheduler` have a branch for dependency-free tasks that bundle an array's chunks:
the task has no dependencies and its FIRST argument is a dict whose values may be DataNodes holding a
:class:`~deisa.ray.types.ChunkRef`. That is how a sliced/decorated chunk array is represented, and the task is
assigned to the actor that owns the referenced chunk.

The branch used to assume, unconditionally, that ``task.args[0]`` was a dict. But array CREATION tasks are also
dependency-free Tasks and their first argument is a scalar or a DataNode:

- ``da.arange(...)`` -> ``Task(key, np.arange, (start, stop, step, ...))``  (``args[0]`` an ``int``)
- ``da.eye(...)``    -> ``args[0]`` an ``int``
- ``da.full(...)``   -> ``args[0]`` a ``DataNode``

Calling ``.items()`` on those raised ``'int' object has no attribute 'items'`` (or the ``DataNode`` variant) whenever
the distributed scheduler partitioned such a graph. The existing tests missed it because every graph they exercised
either used ``da.ones`` (whose shape is a DataNode) or reductions that fused the creation task away, so the branch was
never reached with a non-dict first argument.

These tests call the partitioners directly, on the exact materialized graphs dask hands a scheduler, so they need no
Ray cluster and pin the failure precisely.
"""

from __future__ import annotations

import dask
import dask.array as da
import pytest
from dask._task_spec import DataNode
from dask.local import get_sync

from deisa.ray._scheduler import greedy_partitioning, random_partitioning
from deisa.ray.types import ChunkRef

PARTITIONERS = (greedy_partitioning, random_partitioning)
PARTITIONER_IDS = ("greedy", "random")
ACTORS = {"a": object(), "b": object()}


def _capture_graph(build) -> dict:
    """Return the materialized graph dask would hand to a scheduler when computing ``build()``."""
    captured: dict = {}

    def sched(dsk, keys, **kwargs):
        captured["dsk"] = dsk
        return get_sync(dsk.dask, keys, **kwargs)

    with dask.config.set(scheduler=sched):
        build().compute()
    return captured["dsk"].dask


@pytest.mark.parametrize("partitioner", PARTITIONERS, ids=PARTITIONER_IDS)
@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: da.arange(10, chunks=3), id="arange"),
        pytest.param(lambda: da.eye(3, chunks=2), id="eye"),
        pytest.param(lambda: da.full((4,), 7, chunks=2), id="full"),
        pytest.param(lambda: da.linspace(0, 1, 5, chunks=2), id="linspace"),
    ],
)
def test_partitioner_handles_dependency_free_creation_tasks(partitioner, build) -> None:
    """A creation task with no dependencies and a non-dict first argument must partition, not raise AttributeError."""
    graph = _capture_graph(build)

    assignment = partitioner(dict(graph.items()), dict(ACTORS))

    assert set(assignment) == set(graph), "every task key must be assigned to an actor"
    assert set(assignment.values()) <= set(ACTORS), "assignments must be real actor IDs"


@pytest.mark.parametrize("partitioner", PARTITIONERS, ids=PARTITIONER_IDS)
def test_chunkref_tasks_are_assigned_to_their_owning_actor(partitioner) -> None:
    """The guard must not break the branch it protects: ChunkRef tasks still go to the actor that owns the chunk."""
    ref = object()
    name = "f_0"
    # dask wraps each raw chunk value in a DataNode when it materializes the array's graph, so a real ChunkRef task is
    # a DataNode holding the ChunkRef -- not a bare ChunkRef.
    graph = {
        (name, 0, 0): DataNode((name, 0, 0), ChunkRef(ref=ref, actorid="a", array_name="f", timestep=0, bridge_id=0)),
        (name, 0, 1): DataNode((name, 0, 1), ChunkRef(ref=ref, actorid="b", array_name="f", timestep=0, bridge_id=1)),
    }

    assignment = partitioner(dict(graph.items()), dict(ACTORS))

    assert assignment[(name, 0, 0)] == "a"
    assert assignment[(name, 0, 1)] == "b"


def test_data_node_graph_is_left_alone_for_non_dict_tasks() -> None:
    """Spot-check the concrete shapes: ``da.arange`` args[0] is an int, ``da.full`` args[0] is a DataNode."""
    from dask._task_spec import Task

    arange_graph = _capture_graph(lambda: da.arange(10, chunks=3))
    arange_tasks = [v for v in arange_graph.values() if isinstance(v, Task)]
    assert arange_tasks, "expected at least one Task in the arange graph"
    assert any(isinstance(t.args[0], int) for t in arange_tasks)

    full_graph = _capture_graph(lambda: da.full((4,), 7, chunks=2))
    full_tasks = [v for v in full_graph.values() if isinstance(v, Task)]
    assert full_tasks, "expected at least one Task in the full graph"
    assert any(isinstance(t.args[0], DataNode) for t in full_tasks)
