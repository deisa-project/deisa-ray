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