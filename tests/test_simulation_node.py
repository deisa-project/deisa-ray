# TODO: test clean exit if scheduling actor not created

import concurrent.futures
import ray
import pytest
import numpy as np

from ray.util.state import list_actors
from deisa.ray.types import RayActorHandle
from tests.stubs import StubSchedulingActor

from deisa.ray.bridge import Bridge


def _actor_names_by_prefix(prefix="sched-"):
    actors = list_actors(filters=[("state", "=", "ALIVE")])
    names = []
    for a in actors:
        name = a.get("name")
        ns = a.get("ray_namespace")
        if name and ns == "deisa_ray" and name.startswith(prefix):
            names.append(name)
    return set(names)


arrays_md = {
    "array": {
        "chunk_shape": (1, 1),
        "nb_chunks_per_dim": (1, 1),
        "nb_chunks_of_node": 1,
        "dtype": np.int32,
        "chunk_position": (0, 0),
    }
}


def test_init(ray_cluster, monkeypatch):
    fake_node_id = "FAKE-NODE-1"
    sys_md = {"world_size": 1, "master_address": "127.0.0.1", "master_port": 29500}
    monkeypatch.setattr("deisa.ray.bridge.init_gloo", lambda *a, **k: None)
    monkeypatch.setattr("deisa.ray.bridge.dist.barrier", lambda *a, **k: None)
    c = Bridge(
        bridge_id=0,
        arrays_metadata=arrays_md,
        system_metadata=sys_md,
        _node_id=fake_node_id,
        scheduling_actor_cls=StubSchedulingActor,
    )
    assert c.node_id == fake_node_id
    assert isinstance(c.node_actor, RayActorHandle)
    assert isinstance(c, Bridge)


@pytest.mark.parametrize("nb_nodes", [1, 2, 4])
def test_init_race_free(nb_nodes, ray_cluster, monkeypatch):
    # IMPORTANT: torch.distributed cannot be initialized from multiple threads in one process.
    # This test is about Ray actor init race-freedom, so stub gloo init and dist barrier
    monkeypatch.setattr("deisa.ray.bridge.init_gloo", lambda *a, **k: None)
    monkeypatch.setattr("deisa.ray.bridge.dist.barrier", lambda *a, **k: None)

    ranks_per_node = 10
    fake_node_ids = [f"FAKE-NODE-{n + 1}" for _ in range(ranks_per_node) for n in range(nb_nodes)]

    world_size = len(fake_node_ids)

    def _mk(args):
        rank, node_id = args
        sys_md = {
            "world_size": world_size,
            "master_address": "127.0.0.1",
            "master_port": 29500,
        }
        Bridge(
            bridge_id=rank,  # IMPORTANT: unique rank per simulated process
            arrays_metadata=arrays_md,
            system_metadata=sys_md,
            _node_id=node_id,
            scheduling_actor_cls=StubSchedulingActor,
        )
        return True

    # Start many in parallel (threads are fine; Bridge uses Ray for concurrency)
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
        results = list(ex.map(_mk, list(enumerate(fake_node_ids))))

    assert all(results)

    names = _actor_names_by_prefix()
    for name in [f"FAKE-NODE-{n + 1}" for n in range(nb_nodes)]:
        assert f"sched-{name}" in names
    assert len(names) == nb_nodes


def test_actor_dies_and_client_recovers(ray_cluster, monkeypatch):
    # NOTE: not sure needed because client init happens just once at the beginning.
    monkeypatch.setattr("deisa.ray.bridge.init_gloo", lambda *a, **k: None)
    monkeypatch.setattr("deisa.ray.bridge.dist.barrier", lambda *a, **k: None)
    fake_node_id = "CRASHY-NODE"

    # First client brings up the actor
    sys_md = {"world_size": 1, "master_address": "127.0.0.1", "master_port": 29500}
    Bridge(
        bridge_id=0,
        arrays_metadata=arrays_md,
        system_metadata=sys_md,
        _node_id=fake_node_id,
        scheduling_actor_cls=StubSchedulingActor,
    )
    # Find the actor handle and kill it
    a = ray.get_actor(f"sched-{fake_node_id}", namespace="deisa_ray")
    ray.kill(a, no_restart=True)

    # Now, creating another client should recover (thanks to retry in Bridge.__init__)
    Bridge(
        bridge_id=0,
        arrays_metadata=arrays_md,
        system_metadata=sys_md,
        _node_id=fake_node_id,
        scheduling_actor_cls=StubSchedulingActor,
        _init_retries=5,
    )

    # Also check that a fresh actor exists with the same name
    a2 = ray.get_actor(f"sched-{fake_node_id}", namespace="deisa_ray")
    assert a2 is not None
