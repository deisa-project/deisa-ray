from __future__ import annotations
from typing import Any, Mapping


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
) -> None:
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
    # chunk_shape: tuple/list of positive ints
    chunk_shape = meta["chunk_shape"]
    if not (isinstance(chunk_shape, (tuple, list)) and all(isinstance(n, int) and n > 0 for n in chunk_shape)):
        raise TypeError(
            f"arrays_metadata['{name}']['chunk_shape'] must be a sequence of positive ints, got {chunk_shape!r}"
        )

    # nb_chunks_per_dim: same pattern
    nb_chunks_per_dim = meta["nb_chunks_per_dim"]
    if not (
        isinstance(nb_chunks_per_dim, (tuple, list)) and all(isinstance(n, int) and n > 0 for n in nb_chunks_per_dim)
    ):
        raise TypeError(
            f"arrays_metadata['{name}']['nb_chunks_per_dim'] must be a "
            f"sequence of positive ints, got {nb_chunks_per_dim!r}"
        )

    # nb_chunks_of_node: positive int
    nb_chunks_of_node = meta["nb_chunks_of_node"]
    if not (isinstance(nb_chunks_of_node, int) and nb_chunks_of_node > 0):
        raise TypeError(
            f"arrays_metadata['{name}']['nb_chunks_of_node'] must be a positive int, "
            f"got {type(meta['nb_chunks_of_node']).__name__}"
        )

    # chunk_position: sequence of ints of same length as chunk_shape (optional)
    chunk_position = meta["chunk_position"]
    if not (
        isinstance(chunk_position, (tuple, list))
        and all(
            isinstance(pos, int) and 0 <= pos < nb_chunks for pos, nb_chunks in zip(chunk_position, nb_chunks_per_dim)
        )
    ):
        raise TypeError(
            f"arrays_metadata['{name}']['chunk_position'] must be a sequence of ints, got {chunk_position!r}"
        )

    if len(chunk_position) != len(meta["chunk_shape"]):
        raise ValueError(f"arrays_metadata['{name}']['chunk_position'] must have the same length as 'chunk_shape'")


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

        _validate_single_array_metadata(array_name, meta)
        validated[array_name] = dict(meta)

    return validated
