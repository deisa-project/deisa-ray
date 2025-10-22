# TODO: test clean exit if scheduling actor not created

import concurrent.futures
import ray
import pytest

from ray.util.state import list_actors
from tests.stubs import StubSchedulingActor

from doreisa.simulation_node import Client


def _actor_names_by_prefix(prefix="sched-"):
    actors = list_actors(filters=[("state", "=", "ALIVE")])
    names = []
    for a in actors:
        name = a.get("name")
        ns = a.get("ray_namespace")
        if name and ns == "doreisa" and name.startswith(prefix):
            names.append(name)
    return set(names)


@pytest.mark.parametrize(
    "inpt, inpt_doubled",
    [
        (2, 4),
        (4, 8),
        (6, 12),
    ],
)
def test_stub_actor_basic(ray_cluster, inpt, inpt_doubled):
    a = StubSchedulingActor.options(
        name="stub-alone", namespace="doreisa", lifetime="detached", get_if_exists=True
    ).remote("node-X")
    ref = ray.get(a.preprocessing_callbacks.remote())
    assert isinstance(ref, ray.ObjectRef)
    cbs = ray.get(ref)
    assert isinstance(cbs, dict)
    assert callable(cbs["default"]) and callable(cbs["double"])
    assert cbs["default"](inpt) == inpt and cbs["double"](inpt) == inpt_doubled


def test_init(ray_cluster):
    fake_node_id = "FAKE-NODE-1"
    c = Client(_node_id=fake_node_id, scheduling_actor_cls=StubSchedulingActor)
    assert c.node_id == fake_node_id
    assert isinstance(c.scheduling_actor, ray.actor.ActorHandle)
    assert isinstance(c.preprocessing_callbacks, dict)
    assert callable(c.preprocessing_callbacks["default"]) and callable(c.preprocessing_callbacks["double"])
    assert isinstance(c, Client)


@pytest.mark.parametrize("nb_nodes", [1, 2, 4])
def test_init_race_free(nb_nodes, ray_cluster):
    ranks_per_node = 10  # simulate 100 MPI ranks on same node
    fake_node_ids = [f"FAKE-NODE-{n + 1}" for i in range(ranks_per_node) for n in range(nb_nodes)]

    def _mk(id):
        Client(_node_id=id, scheduling_actor_cls=StubSchedulingActor)
        return True

        # Start many in parallel (threads are fine; Client uses Ray for concurrency)

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
    Client(_node_id=fake_node_id, scheduling_actor_cls=StubSchedulingActor)
    # Find the actor handle and kill it
    a = ray.get_actor(f"sched-{fake_node_id}", namespace="doreisa")
    ray.kill(a, no_restart=True)

    # Now, creating another client should recover (thanks to retry in Client.__init__)
    c2 = Client(_node_id=fake_node_id, scheduling_actor_cls=StubSchedulingActor, _init_retries=5)
    assert isinstance(c2.preprocessing_callbacks, dict)

    # Also check that a fresh actor exists with the same name
    a2 = ray.get_actor(f"sched-{fake_node_id}", namespace="doreisa")
    assert a2 is not None
