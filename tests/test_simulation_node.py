# TODO: test clean exit if scheduling actor not created

import inspect

import pytest

from deisa.ray.bridge import Bridge

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
