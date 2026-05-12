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

    raise TypeError(f"arrays_metadata['{array_name}']['{field_name}'] must be a sequence of ints, got {value!r}")


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

        - ``global_shape``: sequence of positive ints
        - ``chunk_shape``: sequence of positive ints
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

    # global_shape: tuple/list/1d ndarray of positive ints
    global_shape = _normalize_int_sequence(meta["global_shape"], field_name="global_shape", array_name=name)
    if not all(n > 0 for n in global_shape):
        raise TypeError(
            f"arrays_metadata['{name}']['global_shape'] must be a sequence of positive ints, got {global_shape!r}"
        )
    normalized_meta["global_shape"] = global_shape

    # chunk_shape: tuple/list/1d ndarray of positive ints
    chunk_shape = _normalize_int_sequence(meta["chunk_shape"], field_name="chunk_shape", array_name=name)
    if not all(n > 0 for n in chunk_shape):
        raise TypeError(
            f"arrays_metadata['{name}']['chunk_shape'] must be a sequence of positive ints, got {chunk_shape!r}"
        )
    normalized_meta["chunk_shape"] = chunk_shape

    if len(global_shape) != len(chunk_shape):
        raise ValueError(f"arrays_metadata['{name}']['global_shape'] must have the same length as 'chunk_shape'")

    if any(global_dim % chunk_dim != 0 for global_dim, chunk_dim in zip(global_shape, chunk_shape)):
        raise ValueError(
            f"arrays_metadata['{name}']['global_shape'] must be evenly divisible by 'chunk_shape'"
        )

    nb_chunks_per_dim = tuple(global_dim // chunk_dim for global_dim, chunk_dim in zip(global_shape, chunk_shape))

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
        "global_shape",
        "chunk_shape",
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

        extra = meta.keys() - required_keys
        if extra:
            raise ValueError(f"arrays_metadata['{array_name}'] has unknown keys: {extra}")

        validated[array_name] = _validate_single_array_metadata(array_name, meta)

    return validated
