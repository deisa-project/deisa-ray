"""
Integration test: :class:`MergeablePCA` fits the arrays deisa-ray actually delivers.

deisa-ray hands callbacks a :class:`deisa.ray.types.DeisaArray` (a ``dask.array.Array`` subclass carrying a timestep)
assembled from Ray object-store chunks. The estimator is dask-only, so it should accept a ``DeisaArray`` unchanged and
agree with a plain numpy fit.
"""

from __future__ import annotations

import numpy as np
from dask.array import Array as DaskArray

from deisa.ray import MergeablePCA
from deisa.ray.types import DeisaArray


def _deisa_array(data: np.ndarray, chunks, timestep: int) -> DeisaArray:
    """Build a ``DeisaArray`` the way deisa-ray does, without a Ray cluster."""
    import dask.array as da

    base = da.from_array(data, chunks=chunks)
    return DeisaArray(
        base.dask,
        base.name,
        base.chunks,
        t=timestep,
        dtype=base.dtype,
        meta=base._meta,
        shape=base.shape,
    )


def test_fit_accepts_a_deisa_array_and_matches_numpy():
    rng = np.random.RandomState(0)
    data = rng.randn(200, 12)

    on_bridge = _deisa_array(data, chunks=(50, 12), timestep=3)
    assert isinstance(on_bridge, DaskArray)
    assert on_bridge.timestep == 3

    from_deisa = MergeablePCA(n_components=4, whiten=True).fit(on_bridge)
    from_numpy = MergeablePCA(n_components=4, whiten=True).fit(data)

    assert from_deisa.n_features_in_ == 12
    assert from_deisa.n_samples_ == 200
    assert np.allclose(from_deisa.singular_values_, from_numpy.singular_values_)
    assert np.allclose(from_deisa.mean_, data.mean(axis=0))
    assert np.allclose(from_deisa.explained_variance_ratio_, from_numpy.explained_variance_ratio_)


def test_transform_and_inverse_transform_round_trip_on_a_deisa_array():
    rng = np.random.RandomState(1)
    data = rng.randn(120, 8)

    estimator = MergeablePCA(n_components=8).fit(_deisa_array(data, chunks=(40, 8), timestep=0))
    projected = estimator.transform(_deisa_array(data, chunks=(40, 8), timestep=1))
    restored = estimator.inverse_transform(projected)

    # Full rank: the retained subspace is the whole feature space, so the round trip is exact.
    assert np.allclose(np.asarray(restored), data, atol=1e-8)
