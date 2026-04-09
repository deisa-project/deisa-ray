# TODO: test clean exit if scheduling actor not created

import concurrent.futures
import ray
import pytest
import numpy as np
import torch.distributed as dist

from ray.util.state import list_actors
from deisa.ray.types import RayActorHandle
from tests.stubs import StubSchedulingActor
from deisa.ray.bridge import Bridge
from deisa.ray.comm import MPICommAdapter, NoOpComm, TorchDistComm
from tests.utils import pick_free_port


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
    port = pick_free_port()
    sys_md = {"world_size": 1, "master_address": "127.0.0.1", "master_port": port}
    c = Bridge(
        id=0,
        arrays_metadata=arrays_md,
        system_metadata=sys_md,
        _node_id=fake_node_id,
        comm=NoOpComm(0, 1),
        scheduling_actor_cls=StubSchedulingActor,
    )
    assert c.node_id == fake_node_id
    assert isinstance(c.node_actor, RayActorHandle)
    assert isinstance(c, Bridge)


def test_init_with_default_gloo_comm(ray_cluster):
    fake_node_id = "FAKE-NODE-GLOO"
    port = pick_free_port()
    sys_md = {"world_size": 1, "master_address": "127.0.0.1", "master_port": port}

    if dist.is_initialized():
        dist.destroy_process_group()

    try:
        c = Bridge(
            id=0,
            arrays_metadata=arrays_md,
            system_metadata=sys_md,
            _node_id=fake_node_id,
            scheduling_actor_cls=StubSchedulingActor,
        )
        assert isinstance(c.comm, TorchDistComm)
        assert c.comm.rank == 0
        assert c.comm.world_size == 1
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_init_requires_system_metadata_for_default_gloo(ray_cluster):
    fake_node_id = "FAKE-NODE-MISSING-SYS-MD"

    with pytest.raises(ValueError, match="system_metadata is required when comm is None"):
        Bridge(
            id=0,
            arrays_metadata=arrays_md,
            system_metadata=None,
            _node_id=fake_node_id,
            scheduling_actor_cls=StubSchedulingActor,
        )


def test_init_raises_when_comm_and_system_metadata_are_none(ray_cluster):
    fake_node_id = "FAKE-NODE-NO-COMM-NO-SYS-MD"

    with pytest.raises(ValueError, match="system_metadata is required when comm is None"):
        Bridge(
            id=0,
            arrays_metadata=arrays_md,
            system_metadata=None,
            comm=None,
            _node_id=fake_node_id,
            scheduling_actor_cls=StubSchedulingActor,
        )


def test_init_with_mpi_comm_adapter(ray_cluster):
    class FakeMPIComm:
        def __init__(self):
            self.barrier_calls = 0

        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

        def Barrier(self):
            self.barrier_calls += 1

    fake_node_id = "FAKE-NODE-MPI"
    fake_mpi_comm = FakeMPIComm()
    mpi_comm = MPICommAdapter(fake_mpi_comm)

    c = Bridge(
        id=0,
        arrays_metadata=arrays_md,
        system_metadata=None,
        _node_id=fake_node_id,
        comm=mpi_comm,
        scheduling_actor_cls=StubSchedulingActor,
    )

    assert c.comm is mpi_comm
    assert c.comm.rank == 0
    assert c.comm.world_size == 1
    assert fake_mpi_comm.barrier_calls == 1


def test_init_with_raw_mpi_comm(ray_cluster):
    pytest.importorskip("mpi4py")
    from mpi4py import MPI

    fake_node_id = "FAKE-NODE-RAW-MPI"

    c = Bridge(
        id=0,
        arrays_metadata=arrays_md,
        system_metadata=None,
        _node_id=fake_node_id,
        comm=MPI.COMM_SELF,
        scheduling_actor_cls=StubSchedulingActor,
    )

    assert isinstance(c.comm, MPICommAdapter)
    assert c.comm.rank == 0
    assert c.comm.world_size == 1


def test_init_normalizes_list_chunk_metadata(ray_cluster):
    fake_node_id = "FAKE-NODE-LIST-META"
    port = pick_free_port()
    sys_md = {"world_size": 1, "master_address": "127.0.0.1", "master_port": port}
    list_arrays_md = {
        "array": {
            "chunk_shape": [1, 1],
            "nb_chunks_per_dim": [1, 1],
            "nb_chunks_of_node": 1,
            "dtype": np.int32,
            "chunk_position": [0, 0],
        }
    }

    c = Bridge(
        id=0,
        arrays_metadata=list_arrays_md,
        system_metadata=sys_md,
        _node_id=fake_node_id,
        comm=NoOpComm(0, 1),
        scheduling_actor_cls=StubSchedulingActor,
    )

    assert c.arrays_metadata["array"]["chunk_shape"] == (1, 1)
    assert c.arrays_metadata["array"]["nb_chunks_per_dim"] == (1, 1)
    assert c.arrays_metadata["array"]["chunk_position"] == (0, 0)


def test_init_normalizes_ndarray_chunk_metadata(ray_cluster):
    fake_node_id = "FAKE-NODE-NDARRAY-META"
    port = pick_free_port()
    sys_md = {"world_size": 1, "master_address": "127.0.0.1", "master_port": port}
    ndarray_arrays_md = {
        "array": {
            "chunk_shape": np.array([1, 1], dtype=np.int64),
            "nb_chunks_per_dim": np.array([1, 1], dtype=np.int64),
            "nb_chunks_of_node": np.array(1, dtype=np.int64),
            "dtype": np.int32,
            "chunk_position": np.array([0, 0], dtype=np.int64),
        }
    }

    c = Bridge(
        id=0,
        arrays_metadata=ndarray_arrays_md,
        system_metadata=sys_md,
        _node_id=fake_node_id,
        comm=NoOpComm(0, 1),
        scheduling_actor_cls=StubSchedulingActor,
    )

    assert c.arrays_metadata["array"]["chunk_shape"] == (1, 1)
    assert c.arrays_metadata["array"]["nb_chunks_per_dim"] == (1, 1)
    assert c.arrays_metadata["array"]["nb_chunks_of_node"] == 1
    assert isinstance(c.arrays_metadata["array"]["nb_chunks_of_node"], int)
    assert c.arrays_metadata["array"]["chunk_position"] == (0, 0)


def test_close_returns_timestep_and_logs(ray_cluster, caplog):
    fake_node_id = "FAKE-NODE-CLOSE"
    port = pick_free_port()
    sys_md = {"world_size": 1, "master_address": "127.0.0.1", "master_port": port}
    c = Bridge(
        id=0,
        arrays_metadata=arrays_md,
        system_metadata=sys_md,
        _node_id=fake_node_id,
        comm=NoOpComm(0, 1),
        scheduling_actor_cls=StubSchedulingActor,
    )

    with caplog.at_level("INFO", logger="deisa.ray.bridge"):
        last_timestep = c.close(timestep=7)

    assert last_timestep == 7
    assert "Bridge 0 closed at timestep 7" in caplog.text


@pytest.mark.parametrize("nb_nodes", [1, 2, 4])
def test_init_race_free(nb_nodes, ray_cluster):
    # IMPORTANT: torch.distributed cannot be initialized from multiple threads in one process.
    # This test is about Ray actor init race-freedom, so stub gloo init and dist barrier

    ranks_per_node = 10
    port = pick_free_port()
    fake_node_ids = [(f"FAKE-NODE-{n + 1}", port) for _ in range(ranks_per_node) for n in range(nb_nodes)]

    world_size = len(fake_node_ids)

    def _mk(args):
        rank, (node_id, port) = args
        sys_md = {
            "world_size": world_size,
            "master_address": "127.0.0.1",
            "master_port": port,
        }
        Bridge(
            id=rank,  # IMPORTANT: unique rank per simulated process
            arrays_metadata=arrays_md,
            system_metadata=sys_md,
            _node_id=node_id,
            comm=NoOpComm(rank, world_size),
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


def test_actor_dies_and_client_recovers(ray_cluster):
    # NOTE: not sure needed because client init happens just once at the beginning.
    fake_node_id = "CRASHY-NODE"
    port = pick_free_port()

    # First client brings up the actor
    sys_md = {"world_size": 1, "master_address": "127.0.0.1", "master_port": port}
    Bridge(
        id=0,
        arrays_metadata=arrays_md,
        system_metadata=sys_md,
        _node_id=fake_node_id,
        comm=NoOpComm(0, 1),
        scheduling_actor_cls=StubSchedulingActor,
    )
    # Find the actor handle and kill it
    a = ray.get_actor(f"sched-{fake_node_id}", namespace="deisa_ray")
    ray.kill(a, no_restart=True)

    # Now, creating another client should recover (thanks to retry in Bridge.__init__)
    Bridge(
        id=0,
        arrays_metadata=arrays_md,
        system_metadata=sys_md,
        _node_id=fake_node_id,
        scheduling_actor_cls=StubSchedulingActor,
        comm=NoOpComm(0, 1),
        _init_retries=5,
    )

    # Also check that a fresh actor exists with the same name
    a2 = ray.get_actor(f"sched-{fake_node_id}", namespace="deisa_ray")
    assert a2 is not None
