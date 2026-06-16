import time

import dask
import numpy as np
import pytest
import ray

from deisa.ray.config import DEISA_DISTRIBUTED_SCHEDULING_ENV
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from tests.utils import pick_free_port, ray_multinode_cluster, wait_for_head_node  # noqa: F401

# I need to test:
# - [x] that the head actor is queried directly for feedback (not the node actors)
# - [x] that the feedback queue is fixed size
# - [x] rejects non-increasing timesteps
# - [x] that querying a non-existent key returns (False, None)

# - [x] that querying without a timestep returns all feedback in the queue
# - [x] that querying a valid timestep returns a hit (True, value)
# - [x] that querying a timestep that is too old returns a miss (False, None)
# - [x] that querying a timestep that is too new returns a miss (False, None)
# - [x] that feedback is broadcast to all workers, and that workers can retrieve it via the bridge.get method


@pytest.fixture(autouse=True)
def reset_process_state(monkeypatch):
    scheduler = dask.config.get("scheduler", default=None)
    monkeypatch.delenv(DEISA_DISTRIBUTED_SCHEDULING_ENV, raising=False)
    yield
    monkeypatch.delenv(DEISA_DISTRIBUTED_SCHEDULING_ENV, raising=False)
    dask.config.set(scheduler=scheduler)


class _RecordingRemoteMethod:
    def __init__(self, object_ref: str) -> None:
        self.object_ref = object_ref
        self.calls: list[tuple[object, ...]] = []

    def remote(self, *args):
        self.calls.append(args)
        return self.object_ref


class _RecordingHeadActor:
    def __init__(self) -> None:
        self.get_feedback = _RecordingRemoteMethod("head-feedback-ref")


class _FailingRemoteMethod:
    def remote(self, *args):
        raise AssertionError("node actor should not be queried for timestamped feedback")


class _FailingNodeActor:
    def __init__(self) -> None:
        self.get_feedback = _FailingRemoteMethod()


def test_bridge_zero_queries_head_actor_directly(monkeypatch) -> None:
    """
    Test that the Bridge.get method queries the head actor directly for feedback, and not the node actors.
    """
    from deisa.ray.bridge import Bridge
    from deisa.ray.comm import NoOpComm

    bridge = Bridge.__new__(Bridge)
    bridge.bridge_id = 0
    bridge.comm = NoOpComm()
    bridge.head_actor = _RecordingHeadActor()
    bridge.node_actor = _FailingNodeActor()
    bridge._closed = True

    def fake_ray_get(object_ref):
        assert object_ref == "head-feedback-ref"
        return True, "direct"

    monkeypatch.setattr(ray, "get", fake_ray_get)

    assert bridge.get("foo", timestep=7) == "direct"
    assert bridge.head_actor.get_feedback.calls == [("foo", 7)]


def test_bridge_get_returns_default_when_feedback_missing(monkeypatch) -> None:
    from deisa.ray.bridge import Bridge
    from deisa.ray.comm import NoOpComm

    bridge = Bridge.__new__(Bridge)
    bridge.bridge_id = 0
    bridge.comm = NoOpComm()
    bridge.head_actor = _RecordingHeadActor()
    bridge.node_actor = _FailingNodeActor()
    bridge._closed = True

    def fake_ray_get(object_ref):
        assert object_ref == "head-feedback-ref"
        return False, None

    monkeypatch.setattr(ray, "get", fake_ray_get)

    assert bridge.get("foo", timestep=7, default="fallback") == "fallback"
    assert bridge.head_actor.get_feedback.calls == [("foo", 7)]


def test_feedback_queue_is_fixed_size(ray_multinode_cluster) -> None:  # noqa: F811
    """Test that the feedback queue in the head actor is fixed size."""
    from deisa.ray.window_handler import Deisa

    deisa = Deisa(feedback_queue_size=2)

    deisa.set("foo", value="old", timestep=0)
    deisa.set("foo", value="middle", timestep=1)
    deisa.set("foo", value="new", timestep=2)

    # querying "foo" at timestep 0 should miss since it doesnt exist in queue (queue removed bc too old)
    assert ray.get(deisa.head.get_feedback.remote("foo", 0)) == (False, None)
    # querying "foo" at timestep 1 should hit
    assert ray.get(deisa.head.get_feedback.remote("foo", 1)) == (True, "middle")
    # querying "foo" at timestep 2 should hit
    assert ray.get(deisa.head.get_feedback.remote("foo", 2)) == (True, "new")
    # querying "foo" at timestep 3 should miss since its too new (not yet added)
    assert ray.get(deisa.head.get_feedback.remote("foo", 3)) == (False, None)
    # querying "foo" without a timestep should return all feedback in the queue, which should only contain the 2 most recent entries
    assert ray.get(deisa.head.get_feedback.remote("foo")) == (True, [(1, "middle"), (2, "new")])


def test_feedback_queue_behavior(ray_multinode_cluster) -> None:  # noqa: F811
    """
    Test that the feedback queue in the head actor rejects non-increasing timesteps.
    """
    from deisa.ray.window_handler import Deisa

    deisa = Deisa(feedback_queue_size=3)

    deisa.set("foo", value="one", timestep=1)

    with pytest.raises(ValueError, match="already been set"):
        deisa.set("foo", value="repeat", timestep=1)

    with pytest.raises(ValueError, match="older than newest timestep"):
        deisa.set("foo", value="old", timestep=0)

    deisa.set("foo", value="two", timestep=2)
    deisa.set("bar", value="independent", timestep=0)

    # querying "foo" at timestep 0 should miss since it doesnt exist in queue (rejected for being too old)
    assert ray.get(deisa.head.get_feedback.remote("foo", 0)) == (False, None)
    # querying "foo" at timestep 1 should hit
    assert ray.get(deisa.head.get_feedback.remote("foo", 1)) == (True, "one")
    # querying "foo" at timestep 2 should hit
    assert ray.get(deisa.head.get_feedback.remote("foo", 2)) == (True, "two")
    # querying "bar" at timestep 0 should hit since it's independent of "foo" and has its own entry in the queue
    assert ray.get(deisa.head.get_feedback.remote("bar", 0)) == (True, "independent")
    # querying "bar" at timestep 1 should miss since it doesnt exist in queue
    assert ray.get(deisa.head.get_feedback.remote("bar", 1)) == (False, None)
    # querying "joe" at any timestep should miss since it doesnt exist in queue
    assert ray.get(deisa.head.get_feedback.remote("joe", 0)) == (False, None)
    assert ray.get(deisa.head.get_feedback.remote("joe", 1)) == (False, None)
    assert ray.get(deisa.head.get_feedback.remote("joe")) == (False, None)


def test_feedback_set_does_not_leak_dask_scheduler(ray_multinode_cluster) -> None:  # noqa: F811
    """Publishing feedback should not change later unrelated Dask computations."""
    from deisa.ray.window_handler import Deisa

    dask.config.set(scheduler="threads")

    deisa = Deisa(feedback_queue_size=2)
    deisa.set("foo", value="one", timestep=1)

    assert ray.get(deisa.head.get_feedback.remote("foo", 1)) == (True, "one")
    assert dask.config.get("scheduler") == "threads"


@ray.remote(max_retries=0)
def feedback_head() -> bool:
    from deisa.ray.types import DeisaArray, Window
    from deisa.ray.window_handler import Deisa

    deisa = Deisa(feedback_queue_size=2)

    def callback(array: list[DeisaArray]) -> None:
        latest = array[-1]
        if latest.t == 0:
            deisa.set("foo", value=latest.t, timestep=latest.t)
        if latest.t == 1:
            deisa.set(key="foo", value=latest.t, timestep=latest.t)

    deisa.register_callback(callback, *[Window("array")])
    deisa.execute_callbacks()
    return True


@ray.remote(num_cpus=0, max_retries=0)
def feedback_worker(*, rank: int, port: int) -> tuple[int, str, int]:
    from deisa.ray.bridge import Bridge
    from tests.comm_utils import init_gloo_comm

    arrays_md = {
        "array": {
            "global_shape": (1, 2),
            "chunk_shape": (1, 1),
            "chunk_position": (0, rank),
        }
    }

    comm = init_gloo_comm(
        2,
        rank,
        "127.0.0.1",
        port,
    )
    bridge = Bridge(arrays_metadata=arrays_md, comm=comm)

    for timestep in range(2):
        bridge.send(
            array_name="array",
            chunk=np.array([[(rank + 1) * timestep]], dtype=np.int32),
            timestep=timestep,
        )

    feedback_0 = None
    # poll for feedback
    for _ in range(100):
        feedback_0 = bridge.get("foo", timestep=0)
        if feedback_0 is not None:
            break
        time.sleep(0.1)

    missing = bridge.get("foo", timestep=99)
    # This close-before-polling order is intentional. The analytics callback
    # loop flushes a timestep only after it observes a later timestep or this
    # close sentinel, so feedback for the last simulated timestep is not
    # available until the simulation has declared that no more data is coming.
    bridge.close(timestep=2)

    feedback_1 = None
    for _ in range(100):
        feedback_1 = bridge.get("foo", timestep=1)
        if feedback_1 is not None:
            break
        time.sleep(0.1)

    assert feedback_0 is not None
    assert feedback_1 is not None
    assert missing is None
    return feedback_0, missing, feedback_1


def test_bridge_get_broadcasts_timestamped_feedback(ray_multinode_cluster) -> None:  # noqa: F811
    cluster = ray_multinode_cluster["cluster"]
    head_node_id = None
    worker_node_ids = []
    for node in cluster.list_all_nodes():
        if node.is_head():
            head_node_id = node.node_id
        else:
            worker_node_ids.append(node.node_id)

    head_ref = feedback_head.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=head_node_id,
            soft=False,
        ),
    ).remote()
    wait_for_head_node()
    port = pick_free_port()

    worker_refs = [
        feedback_worker.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=worker_node_ids[rank],
                soft=False,
            ),
        ).remote(rank=rank, port=port)
        for rank in range(2)
    ]

    results = ray.get(worker_refs)
    assert ray.get(head_ref)
    assert results == [(0, None, 1), (0, None, 1)]


# Regression note:
# These tests intentionally exercise Deisa.set() from the pytest driver process.
# Before the scheduler scoping fix, Deisa.set() called _ensure_connected(), and
# _ensure_connected() permanently installed Dask-on-Ray as the process-wide Dask
# scheduler. Since test_saving_dask_arrays runs later in the same pytest process,
# its driver-side data.sum().compute() calls inherited that leaked scheduler.
# This was surprising because the feedback loop feature itself does not touch
# HDF5 or Zarr saving, but the shared Dask scheduler config made the two test
# files affect each other. The fix is that window_handler.py now applies the Ray
# scheduler only inside the execute_callbacks() context manager, where Deisa
# actually needs it, instead of changing global Dask config during connection or
# feedback publication.
