# TODO: test clean exit if scheduling actor not created

import inspect
import pytest

from tests.stubs import StubSchedulingActor
from deisa.ray.bridge import Bridge
from deisa.ray.comm import MPICommAdapter, NoOpComm


arrays_md = {
    "array": {
        "global_shape": (1, 1),
        "chunk_shape": (1, 1),
        "chunk_position": (0, 0),
    }
}


def test_init_accepts_variadic_args_signature():
    signature = inspect.signature(Bridge.__init__)
    parameters = list(signature.parameters.values())

    assert [parameter.name for parameter in parameters[:5]] == ["self", "comm", "arrays_metadata", "args", "kwargs"]
    assert parameters[3].kind is inspect.Parameter.VAR_POSITIONAL
    assert parameters[4].kind is inspect.Parameter.VAR_KEYWORD


def test_init_raises_when_comm_is_none():

    with pytest.raises(ValueError, match="comm is required"):
        Bridge(
            arrays_metadata=arrays_md,
            comm=None,
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

    fake_mpi_comm = FakeMPIComm()
    mpi_comm = MPICommAdapter(fake_mpi_comm)

    c = Bridge(
        arrays_metadata=arrays_md,
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

    c = Bridge(
        arrays_metadata=arrays_md,
        comm=MPI.COMM_SELF,
        scheduling_actor_cls=StubSchedulingActor,
    )

    assert isinstance(c.comm, MPICommAdapter)
    assert c.comm.rank == 0
    assert c.comm.world_size == 1


# TODO should be part of deisa.core
# def test_init_normalizes_list_chunk_metadata(ray_cluster):
#     list_arrays_md = {
#         "array": {
#             "global_shape": [1, 1],
#             "chunk_shape": [1, 1],
#             "chunk_position": [0, 0],
#         }
#     }

#     c = Bridge(
#         arrays_metadata=list_arrays_md,
#         comm=NoOpComm(0, 1),
#         scheduling_actor_cls=StubSchedulingActor,
#     )

#     assert c.arrays_metadata["array"]["global_shape"] == [1, 1]
#     assert c.arrays_metadata["array"]["chunk_shape"] == [1, 1]
#     assert "nb_chunks_per_dim" not in c.arrays_metadata["array"]
#     assert c.arrays_metadata["array"]["chunk_position"] == [0, 0]


# TODO should be part of deisa.core
# def test_init_normalizes_ndarray_chunk_metadata(ray_cluster):
#     fake_node_id = "FAKE-NODE-NDARRAY-META"
#     ndarray_arrays_md = {
#         "array": {
#             "global_shape": np.array([1, 1], dtype=np.int64),
#             "chunk_shape": np.array([1, 1], dtype=np.int64),
#             "chunk_position": np.array([0, 0], dtype=np.int64),
#         }
#     }

#     c = Bridge(
#         arrays_metadata=ndarray_arrays_md,
#         comm=NoOpComm(0, 1),
#         _node_id=fake_node_id,
#         scheduling_actor_cls=StubSchedulingActor,
#     )

#     assert c.arrays_metadata["array"]["global_shape"] == (1, 1)
#     assert c.arrays_metadata["array"]["chunk_shape"] == (1, 1)
#     assert "nb_chunks_per_dim" not in c.arrays_metadata["array"]
#     assert c.arrays_metadata["array"]["chunk_position"] == (0, 0)


# TODO should be part of deisa.core
# def test_arrays_metadata_global_shape_must_match_chunk_grid():
#     arrays_md = {
#         "array": {
#             "global_shape": (3, 2),
#             "chunk_shape": (2, 1),
#             "chunk_position": (0, 0),
#         }
#     }

#     with pytest.raises(ValueError, match="evenly divisible"):
#         _validate_arrays_meta(arrays_md)


def test_close_returns_none_and_logs(ray_cluster, caplog):
    c = Bridge(
        arrays_metadata=arrays_md,
        comm=NoOpComm(0, 1),
        scheduling_actor_cls=StubSchedulingActor,
    )

    with caplog.at_level("INFO", logger="deisa.ray.bridge"):
        result = c.close(timestep=7)

    assert result is None
    assert "Bridge 0 closed at timestep 7" in caplog.text