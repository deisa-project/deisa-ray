from __future__ import annotations
from typing import Any, Mapping

import numpy as np


def _normalize_int_sequence(value: Any, *, field_name: str, array_name: str) -> tuple[int, ...]:
    """
    Normalize tuple-like integer metadata to a tuple of Python ints.

    Parameters
    ----------
    value : Any
        User-provided metadata value.
    field_name : str
        Name of the metadata field being normalized.
    array_name : str
        Name of the array owning the metadata.

    Returns
    -------
    tuple[int, ...]
        Normalized tuple of Python integers.

    Raises
    ------
    TypeError
        If ``value`` is not a tuple-like sequence of integers.
    """
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise TypeError(
                f"arrays_metadata['{array_name}']['{field_name}'] must be a 1D sequence of ints, got {value!r}"
            )
        return tuple(value.tolist())

    if isinstance(value, (tuple, list)):
        return tuple(value)

    raise TypeError(
        f"arrays_metadata['{array_name}']['{field_name}'] must be a sequence of ints, got {value!r}"
    )


def _validate_system_meta(system_meta: Mapping[str, Any]) -> dict[str, Any]:
    """
    Validate and normalize the ``system_metadata`` argument.

    Parameters
    ----------
    system_meta : Mapping[str, Any]
        User-provided system-level metadata.

    Returns
    -------
    dict[str, Any]
        A shallow-copied version of the input mapping.

    Raises
    ------
    TypeError
        If ``system_meta`` is not a mapping.
    """
    if not isinstance(system_meta, Mapping):
        raise TypeError(f"system_metadata must be a mapping, got {type(system_meta).__name__}")
    return dict(system_meta)


def _validate_single_array_metadata(
    name: str,
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Validate metadata for a single array entry.

    Parameters
    ----------
    name : str
        Array name.
    meta : Mapping[str, Any]
        Metadata for this array. Must contain at least:

        - ``chunk_shape``: sequence of positive ints
        - ``nb_chunks_per_dim``: sequence of positive ints
        - ``nb_chunks_of_node``: positive int
        - ``dtype``: NumPy dtype or anything accepted by ``np.dtype``
        - ``chunk_position``: sequence of ints of same length as
          ``chunk_shape``

    Raises
    ------
    TypeError
        If any field has an invalid type.
    ValueError
        If shapes/positions have inconsistent lengths.
    """
    normalized_meta = dict(meta)

    # chunk_shape: tuple/list/1d ndarray of positive ints
    chunk_shape = _normalize_int_sequence(meta["chunk_shape"], field_name="chunk_shape", array_name=name)
    if not all(n > 0 for n in chunk_shape):
        raise TypeError(
            f"arrays_metadata['{name}']['chunk_shape'] must be a sequence of positive ints, got {chunk_shape!r}"
        )
    normalized_meta["chunk_shape"] = chunk_shape

    # nb_chunks_per_dim: same pattern
    nb_chunks_per_dim = _normalize_int_sequence(
        meta["nb_chunks_per_dim"],
        field_name="nb_chunks_per_dim",
        array_name=name,
    )
    if not all(n > 0 for n in nb_chunks_per_dim):
        raise TypeError(
            f"arrays_metadata['{name}']['nb_chunks_per_dim'] must be a "
            f"sequence of positive ints, got {nb_chunks_per_dim!r}"
        )
    normalized_meta["nb_chunks_per_dim"] = nb_chunks_per_dim

    # nb_chunks_of_node: positive int
    try:
        nb_chunks_of_node = int(meta["nb_chunks_of_node"])
        assert nb_chunks_of_node > 0
    except:
        raise TypeError(
            f"arrays_metadata['{name}']['nb_chunks_of_node'] must be a positive int, "
            f"got {type(meta['nb_chunks_of_node']).__name__}"
        )
    normalized_meta["nb_chunks_of_node"] = nb_chunks_of_node


    # chunk_position: sequence of ints of same length as chunk_shape
    chunk_position = _normalize_int_sequence(
        meta["chunk_position"],
        field_name="chunk_position",
        array_name=name,
    )
    if not all(0 <= pos < nb_chunks for pos, nb_chunks in zip(chunk_position, nb_chunks_per_dim)):
        raise TypeError(
            f"arrays_metadata['{name}']['chunk_position'] must be a sequence of ints, got {chunk_position!r}"
        )
    normalized_meta["chunk_position"] = chunk_position

    if len(chunk_position) != len(chunk_shape):
        raise ValueError(f"arrays_metadata['{name}']['chunk_position'] must have the same length as 'chunk_shape'")

    return normalized_meta


def _validate_arrays_meta(
    arrays_metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Validate and normalize the ``arrays_metadata`` argument.

    Parameters
    ----------
    arrays_metadata : Mapping[str, Mapping[str, Any]]
        User-provided metadata for all arrays handled by this bridge.

    Returns
    -------
    dict[str, dict[str, Any]]
        A shallow-copied and validated version of the input mapping.

    Raises
    ------
    TypeError
        If the top-level mapping, keys or values have incorrect types.
    ValueError
        If required keys are missing for any array.
    """
    if not isinstance(arrays_metadata, Mapping):
        raise TypeError(f"arrays_metadata must be a mapping from str to dict, got {type(arrays_metadata).__name__}")

    required_keys = {
        "chunk_shape",
        "nb_chunks_per_dim",
        "nb_chunks_of_node",
        "dtype",
        "chunk_position",
    }

    validated: dict[str, dict[str, Any]] = {}

    for array_name, meta in arrays_metadata.items():
        # key type
        if not isinstance(array_name, str):
            raise TypeError(f"arrays_metadata keys must be str, got {type(array_name).__name__}")

        # value type
        if not isinstance(meta, Mapping):
            raise TypeError(f"arrays_metadata['{array_name}'] must be a mapping, got {type(meta).__name__}")

        # required keys present?
        missing = required_keys - meta.keys()
        if missing:
            raise ValueError(f"arrays_metadata['{array_name}'] is missing required keys: {missing}")

        validated[array_name] = _validate_single_array_metadata(array_name, meta)

    return validated
