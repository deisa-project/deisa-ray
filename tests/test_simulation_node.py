# TODO: test clean exit if scheduling actor not created

import concurrent.futures
import ray
import pytest
import numpy as np

from ray.util.state import list_actors
from deisa.ray.types import RayActorHandle
from tests.stubs import StubSchedulingActor

from deisa.ray.bridge import Bridge
from deisa.ray.utils import get_system_metadata


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


def test_init(ray_cluster):
    fake_node_id = "FAKE-NODE-1"
    sys_md = get_system_metadata()
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
def test_init_race_free(nb_nodes, ray_cluster):
    ranks_per_node = 10  # simulate 100 MPI ranks on same node
    fake_node_ids = [f"FAKE-NODE-{n + 1}" for i in range(ranks_per_node) for n in range(nb_nodes)]

    def _mk(id):
        sys_md = get_system_metadata()
        Bridge(
            bridge_id=0,
            arrays_metadata=arrays_md,
            system_metadata=sys_md,
            _node_id=id,
            scheduling_actor_cls=StubSchedulingActor,
        )
        return True

        # Start many in parallel (threads are fine; Bridge uses Ray for concurrency)

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
        results = list(ex.map(_mk, fake_node_ids))

    assert all(results)

    names = _actor_names_by_prefix()
    for name in [f"FAKE-NODE-{n + 1}" for n in range(nb_nodes)]:
        assert f"sched-{name}" in names

    assert len(names) == nb_nodes


def test_actor_dies_and_client_recovers(ray_cluster):
    # NOTE: not sure needed because client init happens just once at the beginning.
    fake_node_id = "CRASHY-NODE"

    # First client brings up the actor
    sys_md = get_system_metadata()
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
