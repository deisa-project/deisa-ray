# =============================================================================
# Copyright (C) 2026 Commissariat a l'energie atomique et aux energies alternatives (CEA)
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the names of CEA, nor the names of the contributors may be used
#   to endorse or promote products derived from this software without specific
#   prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# =============================================================================
"""
Tests for the MergeablePCA Dask estimator: exact fit, local_rank trade-off, graph shape, and refusals.

Two reference implementations, and why both
---------------------------------------------
The estimator is checked against scikit-learn's ``PCA`` wherever scikit-learn is importable, and against a
numpy-only SVD reference everywhere else. The card that introduced the reference ("numpy SVD reference - sklearn is
NOT available") predates scikit-learn 1.9.1 landing in the venv; using the real library when present is strictly
stronger than a hand-rolled reference, so both live here:

- ``sklearn_pca(...)`` guards every sklearn-only test, so the suite still passes in an environment without
  scikit-learn, exactly as the card requires. The import is INSIDE the test, not at module level: a module-level
  ``importorskip`` would skip this whole file, silently disabling every exactness assertion below it.
- ``batch_pca`` below is the numpy-only fallback and is always used for the core exactness assertions, so the
  load-bearing tests never silently skip when a dependency is missing.

Metrics
-------
Sign invariance, again. Eigenvector sign is arbitrary, so components are NEVER compared elementwise. Every accuracy
assertion uses one of:

- **M1** ``subspace_distance(a, b, k)``: ``1 - min singular value`` of the overlap of the two leading ``k``-dimensional
  row subspaces. 0 means "same subspace", independent of signs and of any rotation inside the subspace. Both operands
  are cut to the SAME ``k``; every call site pins ``k < n_features``, because two ``d``-dimensional subspaces of
  ``R**d`` are trivially the whole space and the metric would be vacuous.
- **M2** ``rel_var_error(sv_summary, sv_ref_full)``: relative error of the retained total variance against the
  FULL-rank batch reference. This is the metric for truncation: it falls monotonically as ``local_rank`` grows.
- **max_abs_error``: only ever on sign-free quantities (singular values, means, explained variances).

Why the "no compute" assertions are structural rather than observational
-------------------------------------------------------------------------
The regression this file guards is the export's visualize bug: ``compute()`` called inside the graph builder, so the
graph could not be drawn. Re-testing it by watching for scheduler activity is inherently racy. Instead the tripwire is
DETERMINISTIC: the array's blocks are produced by a ``map_blocks`` function with a module-level side effect, and
``_fit_dask_delayed`` is called while a counter is live. If the builder computed anything, the counter would move. It
is reset to zero right after the array is built, because dask touches blocks once while inferring ``meta``.
"""

from __future__ import annotations

import dask.array as da
import dask.config
import numpy as np
import pytest
from dask.delayed import Delayed

from deisa.ray.mergeable_pca import MergeablePCA, PCASummary


@pytest.fixture(autouse=True)
def default_scheduler():
    """Pin the Dask scheduler for this module, so a leak from another test cannot break it.

    A ``Client`` constructed by another test in the same xdist worker leaves dask's GLOBAL scheduler set to
    ``dask.distributed``. Once that client is gone, a bare ``.compute()`` raises "Requested dask.distributed
    scheduler but no Client active" -- an order-dependent failure that lands on whichever test this worker runs next,
    which under ``-n 16 --dist loadgroup`` is how 37 of these tests failed for a reason unrelated to the estimator.
    test_chain.py hits the same leak and works around it per call site; pinning it once per module is cheaper and
    covers every ``.compute()`` here, including the ones inside the estimator itself.

    Restored afterwards: this is a fix for the TEST environment, not a behaviour change, and it must not leak back.
    """
    previous = dask.config.get("scheduler", default=None)
    dask.config.set(scheduler="threads")
    try:
        yield
    finally:
        dask.config.set(scheduler=previous)


# sklearn and dask-ml are BASELINE comparisons only. Neither is imported by src/, so the library's runtime dependency
# set is unchanged. They are imported LAZILY, inside the tests that need them, via the two factories below.
#
# Deliberately NOT a module-level ``pytest.importorskip``: that skips the ENTIRE file when the library is absent, which
# would silently switch off every load-bearing exactness assertion in it -- the exact failure mode this file is written
# to avoid. Only the reference-comparison tests may skip, and each one skips on its own.


def sklearn_pca(**kwargs):
    """Instantiate scikit-learn's ``PCA``, or skip the CALLING test when scikit-learn is not installed.

    - ``:param kwargs:`` Forwarded to ``sklearn.decomposition.PCA``.
    """
    return pytest.importorskip("sklearn.decomposition", reason="scikit-learn is the reference PCA").PCA(**kwargs)


def dask_ml_incremental_pca(**kwargs):
    """Instantiate dask-ml's ``IncrementalPCA``, or skip the CALLING test when dask-ml is not installed.

    - ``:param kwargs:`` Forwarded to ``dask_ml.decomposition.IncrementalPCA``.
    """
    return pytest.importorskip(
        "dask_ml.decomposition", reason="dask-ml is only an optional baseline comparison"
    ).IncrementalPCA(**kwargs)


# ------------------------------------------------------------------------------ tripwire array
# ------------------------------------------------------------------------------
COMPUTED_BLOCKS: list[int] = []
"""Appended to by :func:`counting_block`; a non-empty list means a block was materialized."""


@pytest.fixture(autouse=True)
def _tripwire_isolated():
    """Clear the module-global tripwire BEFORE and AFTER every test in this module.

    ``COMPUTED_BLOCKS`` is plain module state, and not every ``counting_array`` consumer
    drains it (a test that legitimately computes — e.g. a payload-size or fit-refusal
    test — leaves stale ``1``s behind). Any later "nothing was computed" assertion would
    then read the PREVIOUS test's blocks, which is exactly the cross-test leak that made
    ``test_batch_size_creates_more_leaves_than_row_chunks`` fail on CI whenever
    pytest-xdist happened to schedule the two tests onto the same worker. Isolating the
    tripwire per test makes every such assertion about that test's own graph only.
    """
    COMPUTED_BLOCKS.clear()
    yield
    COMPUTED_BLOCKS.clear()


def counting_block(block):
    """Materialize one block and record it, so "the graph was not computed" is observable deterministically.

    - ``:param block:`` One 2-D block handed over by ``da.map_blocks``.
    """
    COMPUTED_BLOCKS.append(1)
    return np.asarray(block, dtype=np.float64)


def counting_array(n_blocks: int, rows: int = 100, features: int = 8):
    """Build a dask array whose every block records its own materialization, and arm the tripwire.

    The counter is zeroed after construction so that dask's one-time ``meta`` inference does not count as a compute.

    - ``:param n_blocks:`` Number of row chunks, and therefore of leaves the estimator will build.
    - ``:param rows:`` Rows per row chunk.
    - ``:param features:`` Number of features, all in one chunk so the precondition holds by construction.
    """
    COMPUTED_BLOCKS.clear()
    # Built from from_array on purpose: wrapping da.random.random in map_blocks collapses the chunk structure (the
    # resulting array comes back as a single chunk whatever the chunk size), which would silently turn every graph
    # shape test into a one-block test.
    data = np.random.RandomState(0).randn(rows * n_blocks, features)
    X = da.from_array(data, chunks=(rows, features)).map_blocks(counting_block, dtype=np.float64)
    assert X.to_delayed().ravel().size == n_blocks, f"expected {n_blocks} blocks, got {X.chunks}"
    COMPUTED_BLOCKS.clear()  # dask touches blocks once for meta; that is not our graph running.
    return X


def count_tasks(graph) -> dict[str, int]:
    """Count graph tasks by the name of the function they call.

    - ``:param graph:`` A ``dask.delayed.Delayed`` or ``dask`` HighLevelGraph.
    """
    counts: dict[str, int] = {}
    for task in graph.dask.values():
        func = getattr(task, "func", None)
        name = getattr(func, "__name__", None) or type(task).__name__
        counts[name] = counts.get(name, 0) + 1
    return counts


# ------------------------------------------------------------------------------ data and references
# ------------------------------------------------------------------------------
def make_data(n_samples: int, n_features: int, seed: int = 0) -> np.ndarray:
    """Reproducible data; a fixed RandomState keeps every number in this file stable."""
    return np.random.RandomState(seed).randn(n_samples, n_features)


def batch_pca(X: np.ndarray) -> PCASummary:
    """Batch PCA reference on the full array: center everything, then one SVD. numpy only, no sklearn."""
    Xc = np.asarray(X, dtype=np.float64)
    mean = Xc.mean(axis=0)
    _, singular_values, components = np.linalg.svd(Xc - mean, full_matrices=False)
    return PCASummary(
        n_samples=int(Xc.shape[0]),
        mean=mean,
        components=components,
        singular_values=singular_values,
    )


def make_dask_array(n_samples: int = 4096, n_features: int = 50, rows: int = 256):
    """A dask array over a fixed random dataset, features in one chunk, row axis split into ``rows``-sized chunks.

    - ``:param n_samples:`` Rows of the underlying dataset.
    - ``:param n_features:`` Columns, all in one chunk.
    - ``:param rows:`` Rows per Dask chunk, i.e. leaves per tree level.
    """
    return da.from_array(make_data(n_samples, n_features), chunks=(rows, n_features))


# ------------------------------------------------------------------------------ sign-invariant metrics
# ------------------------------------------------------------------------------
def subspace_distance(components_a: np.ndarray, components_b: np.ndarray, k: int) -> float:
    """M1. Distance between the leading ``k``-dimensional row subspaces of two component sets.

    ``1 - min singular value`` of the overlap of the two orthonormal bases. Range ``[0, 1]``, 0 = identical subspace.
    Sign invariant, and invariant to any orthogonal change of basis inside the retained subspace.

    ``k`` MUST be strictly smaller than the feature dimension, or the metric is vacuous: two ``d``-dimensional
    subspaces of ``R**d`` are the whole space, so they are always "identical" no matter how wrong either one is.

    - ``:param components_a:`` Component rows of the first estimator.
    - ``:param components_b:`` Component rows of the second.
    - ``:param k:`` Common retained dimension, strictly less than the feature dimension.
    """
    if k >= components_a.shape[1]:
        raise AssertionError(f"subspace_distance needs k < n_features, got k={k}, n_features={components_a.shape[1]}")
    qa, _ = np.linalg.qr(components_a[:k].T)
    qb, _ = np.linalg.qr(components_b[:k].T)
    return float(1.0 - np.linalg.svd(qa.T @ qb, compute_uv=False).min())


def rel_var_error(singular_values: np.ndarray, singular_values_ref: np.ndarray) -> float:
    """M2. Relative error of the retained total variance, against the FULL-rank batch reference.

    ``singular_values_ref`` must be untruncated, otherwise this measures the truncation itself.

    - ``:param singular_values:`` Singular values under test, possibly truncated.
    - ``:param singular_values_ref:`` Untruncated singular values of the batch PCA.
    """
    total_ref = float(np.sum(np.asarray(singular_values_ref) ** 2))
    return float(abs(float(np.sum(np.asarray(singular_values) ** 2)) - total_ref) / total_ref)


def max_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    """Max absolute elementwise error; only ever used on sign-free quantities (singular values, means).

    - ``:param a:`` First array.
    - ``:param b:`` Second array.
    """
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


# =============================================================================
# End-to-end exactness
# =============================================================================
def test_fit_reproduces_batch_pca_on_a_dask_array():
    """A full-local-rank fit over a dask array IS the batch PCA, up to roundoff. Sign-invariant on the subspace (M1)."""
    X = make_dask_array()
    reference = batch_pca(np.asarray(X))

    pca = MergeablePCA(n_components=10).fit(X)

    assert pca.n_samples_ == 4096
    assert pca.n_features_in_ == 50
    assert pca.n_components_ == 10
    # M2-adjacent: singular values and mean are sign-free, so they compare directly.
    assert max_abs_error(pca.singular_values_, reference.singular_values[:10]) < 1e-10
    assert max_abs_error(pca.mean_, reference.mean) < 1e-12
    # M1 at k = 6 < d = 50: the principal subspace is the batch one.
    assert subspace_distance(pca.components_, reference.components, k=6) < 1e-10
    # And the internal summary carries the whole batch spectrum, not the truncated one.
    assert max_abs_error(pca._summary_.singular_values, reference.singular_values) < 1e-10


@pytest.mark.parametrize("n_blocks", [2, 4, 8, 16, 32])
def test_exactness_holds_for_every_block_count(n_blocks):
    """Exactness is a property of the merge, not of a particular chunking: any block count reproduces the batch PCA.

    Metric M1, the sign-invariant subspace distance, is the accuracy assertion here.
    """
    # The chunk size is DERIVED from n_blocks (not fixed), so the parametrization actually varies the block count.
    total_rows = 32 * 128
    rows = total_rows // n_blocks
    data = make_data(total_rows, 50)
    X = da.from_array(data, chunks=(rows, 50))
    assert X.to_delayed().size == n_blocks
    reference = batch_pca(data)

    pca = MergeablePCA(n_components=5).fit(X)

    assert max_abs_error(pca.singular_values_, reference.singular_values[:5]) < 1e-10
    assert max_abs_error(pca.mean_, reference.mean) < 1e-12
    # M1 at k = 5 < d = 50.
    assert subspace_distance(pca.components_, reference.components, k=5) < 1e-10


def test_fit_matches_sklearn_pca_reference():
    """The library's own reference agrees with sklearn's PCA, componentwise in the sign-invariant sense (M1)."""
    data = make_data(2000, 30)
    reference = sklearn_pca(n_components=8).fit(data)

    pca = MergeablePCA(n_components=8).fit(da.from_array(data, chunks=(250, 30)))

    assert max_abs_error(pca.singular_values_, reference.singular_values_) < 1e-10
    assert max_abs_error(pca.mean_, reference.mean_) < 1e-12
    assert max_abs_error(pca.explained_variance_, reference.explained_variance_) < 1e-10
    # The ratio denominator is the TOTAL pooled variance, so the two agree and the sum is <= 1.
    assert max_abs_error(pca.explained_variance_ratio_, reference.explained_variance_ratio_) < 1e-10
    # M1 at k = 8 < d = 30.
    assert subspace_distance(pca.components_, reference.components_, k=8) < 1e-10


def test_mergeable_pca_is_closer_to_batch_than_dask_ml_incremental_pca():
    """The reason this estimator exists: it is exact, where a sequential truncation-based fit is not.

    dask-ml's ``IncrementalPCA`` truncates its state as it goes, so its singular values are biased low. At full local
    rank this estimator must be closer to the batch reference than dask-ml is. This is the card's "BASELINE
    comparison" made into an assertion, and it is guarded so an environment without dask-ml skips rather than fails.
    """
    data = make_data(2000, 30)
    reference = batch_pca(data).singular_values

    ours = MergeablePCA(n_components=8).fit(da.from_array(data, chunks=(250, 30)))
    incremental = dask_ml_incremental_pca(n_components=8).fit(da.from_array(data, chunks=(250, 30)))

    ours_error = float(np.max(np.abs(ours.singular_values_ - reference[:8])))
    incremental_error = float(np.max(np.abs(np.asarray(incremental.singular_values_) - reference[:8])))
    assert ours_error < 1e-10, f"full local rank should be exact, got {ours_error}"
    assert ours_error < incremental_error, (
        f"mergeable PCA should beat sequential truncation: {ours_error} vs {incremental_error}"
    )


def test_fit_on_a_numpy_array_takes_the_in_memory_path():
    """A numpy input is the single-leaf base case of the same tree, and is just as exact."""
    data = make_data(500, 20)
    reference = batch_pca(data)

    pca = MergeablePCA(n_components=6).fit(data)

    assert pca.n_samples_ == 500
    assert max_abs_error(pca.singular_values_, reference.singular_values[:6]) < 1e-10
    assert max_abs_error(pca.mean_, reference.mean) < 1e-12
    # M1 at k = 6 < d = 20.
    assert subspace_distance(pca.components_, reference.components, k=6) < 1e-10


def test_fit_does_not_mutate_the_input():
    """``copy`` is parity-only and does nothing, because ``fit`` never writes to ``X``."""
    data = make_data(300, 12)
    before = data.copy()

    MergeablePCA(n_components=3).fit(da.from_array(data, chunks=(100, 12)))

    assert np.array_equal(data, before)


def test_n_components_none_keeps_the_full_rank():
    """``n_components=None`` keeps all ``min(n_samples, n_features)`` components."""
    X = make_dask_array(n_samples=1024, n_features=50, rows=256)
    reference = batch_pca(np.asarray(X))

    pca = MergeablePCA().fit(X)

    assert pca.n_components_ == 50
    assert pca.components_.shape == (50, 50)
    assert max_abs_error(pca.singular_values_, reference.singular_values) < 1e-10
    # M1 at k = 25 < d = 50.
    assert subspace_distance(pca.components_, reference.components, k=25) < 1e-10


# =============================================================================
# Internal summary vs public attributes
# =============================================================================
def test_internal_summary_stays_mergeable_and_higher_rank():
    """The mergeable summary is NOT collapsed into the truncated public form. This is what the bridge will emit."""
    X = make_dask_array()

    pca = MergeablePCA(n_components=10).fit(X)

    assert pca._summary_.rank == 50, "the summary must keep the full merged rank, not the requested rank"
    assert pca._summary_.rank > pca.n_components_
    assert pca.singular_values_.shape == (10,)
    assert pca.components_.shape == (10, 50)
    # Unwhitened public components are exactly the leading rows of the summary.
    assert max_abs_error(pca.components_, pca._summary_.components[:10]) < 1e-15
    assert max_abs_error(pca.singular_values_, pca._summary_.singular_values[:10]) < 1e-15
    # The mergeable form is intact: A.T @ A is the pooled centered scatter.
    summary_a = pca._summary_.singular_values[:, None] * pca._summary_.components
    data = np.asarray(X)
    scatter = (data - pca.mean_).T @ (data - pca.mean_)
    assert max_abs_error(summary_a.T @ summary_a, scatter) < 1e-8


def test_summary_is_never_truncated_by_local_rank_when_public_rank_is_smaller():
    """A truncated PUBLIC rank must not leak into the internal summary, or the summary stops being mergeable."""
    X = make_dask_array()

    pca = MergeablePCA(n_components=3).fit(X)

    assert pca._summary_.rank == 50
    assert pca.n_components_ == 3
    assert max_abs_error(pca.singular_values_, pca._summary_.singular_values[:3]) < 1e-15


# =============================================================================
# local_rank trade-off
# =============================================================================
@pytest.mark.parametrize("local_rank", [5, 10, 20, 40, None])
def test_local_rank_degrades_monotonically_and_full_rank_is_exact(local_rank):
    """M1 and M2 both degrade monotonically in ``local_rank``; full local rank is exact on both.

    Monotonicity across the sweep is asserted in :func:`test_local_rank_degradation_is_monotone`; this test pins the
    exactness endpoint and the fact that every truncated rank keeps the public rank.
    """
    X = make_dask_array()
    reference = batch_pca(np.asarray(X))

    pca = MergeablePCA(n_components=10, local_rank=local_rank).fit(X)

    assert pca.n_components_ == 10
    # M1 at k = 10 < d = 50 against the batch subspace.
    assert subspace_distance(pca.components_, reference.components, k=10) < 1.0
    if local_rank is None:
        # Full local rank is exact: both honest metrics sit at roundoff.
        assert max_abs_error(pca.singular_values_, reference.singular_values[:10]) < 1e-10
        assert subspace_distance(pca.components_, reference.components, k=10) < 1e-10
        # M2: no variance lost at full rank.
        assert rel_var_error(pca._summary_.singular_values, reference.singular_values) < 1e-12


def test_local_rank_degradation_is_monotone():
    """As ``local_rank`` grows, the subspace metric (M1) and the variance error (M2) both fall monotonically.

    The slack absorbs float noise when two consecutive ranks yield the same summary; it is many orders of magnitude
    below the smallest measured step, so it cannot hide a real regression. This is the assertion a raw per-component
    error cannot support: that metric is non-monotonic and reads ~2.0 (maximally different) at exact reconstruction.
    """
    X = make_dask_array()
    reference = batch_pca(np.asarray(X))

    ranks = (5, 10, 20, 40, None)
    subspace_distances = []
    variance_errors = []
    for local_rank in ranks:
        pca = MergeablePCA(n_components=10, local_rank=local_rank).fit(X)
        # M1: both sides cut to the same k = 10 < d = 50.
        subspace_distances.append(subspace_distance(pca.components_, reference.components, k=10))
        # M2: retained variance of the summary against the FULL-rank batch reference.
        variance_errors.append(rel_var_error(pca._summary_.singular_values, reference.singular_values))

    slack = 1e-12
    for previous, current in zip(subspace_distances, subspace_distances[1:]):
        assert current <= previous + slack, f"M1 not monotone at ranks={ranks}: {subspace_distances}"
    for previous, current in zip(variance_errors, variance_errors[1:]):
        assert current <= previous + slack, f"M2 not monotone at ranks={ranks}: {variance_errors}"

    # Full local rank is exact on both; the smallest rank is measurably worse than the largest truncated one.
    assert variance_errors[-1] < 1e-10
    assert subspace_distances[-1] < 1e-10
    assert variance_errors[0] > variance_errors[-2]
    assert subspace_distances[0] > subspace_distances[-2]


def test_local_rank_above_the_feature_dim_is_exact():
    """``local_rank >= d`` truncates nothing at the leaves, so the merge is exact."""
    X = make_dask_array()
    reference = batch_pca(np.asarray(X))

    pca = MergeablePCA(n_components=10, local_rank=60).fit(X)

    assert pca._summary_.rank == 50
    assert max_abs_error(pca.singular_values_, reference.singular_values[:10]) < 1e-10
    assert subspace_distance(pca.components_, reference.components, k=10) < 1e-10


def test_local_rank_below_n_components_succeeds_when_the_root_rank_is_enough():
    """``local_rank < n_components`` is allowed as long as the merged rank climbs high enough by the root.

    The merged rank grows with tree depth because each merge adds a correction row, so a small local rank can still
    satisfy a larger ``n_components`` by the root. What must never happen is silently returning FEWER components than
    requested -- that is the refusal :func:`test_fit_refuses_a_root_summary_too_small_for_n_components` pins.

    The result is NOT accurate here, and this test says so explicitly: three retained directions per leaf is a drastic
    truncation, so the merged spectrum satisfies the rank request while the retained subspace stays far from the batch
    one. Asserting that gap is what keeps the rank guard from being mistaken for an accuracy guarantee.
    """
    X = make_dask_array()
    reference = batch_pca(np.asarray(X))

    pca = MergeablePCA(n_components=10, local_rank=3).fit(X)

    assert pca.n_components_ == 10, "the request is satisfied in full, not silently short-changed"
    assert pca._summary_.rank >= 10, "the root rank climbed past n_components, which is why this is allowed at all"
    # M1 at k = 10 < d = 50: measurably wrong, and deliberately asserted as such rather than papered over.
    assert subspace_distance(pca.components_, reference.components, k=10) > 1e-3


def test_local_rank_payload_shrinks_as_local_rank_drops():
    """The bandwidth side of the trade-off: the per-leaf payload is what crosses the process boundary.

    A leaf payload is ``rank * n_features + n_features + rank`` float64 elements, so a smaller ``local_rank`` is a
    strictly smaller summary. This asserts the direction of the knob, not a magic number.
    """
    X = counting_array(n_blocks=1, rows=400)

    sizes = []
    for local_rank in (40, 20, 10, 4):
        summary = MergeablePCA(n_components=4, local_rank=local_rank)._fit_dask_delayed(X).compute()
        sizes.append(int(summary.components.size + summary.mean.size + summary.singular_values.size))

    assert sizes == sorted(sizes, reverse=True), f"payload must shrink with local_rank, got {sizes}"
    assert sizes[0] > sizes[-1]


# =============================================================================
# The delayed graph
# =============================================================================
@pytest.mark.parametrize(
    ("n_blocks", "n_local", "n_merge"),
    [
        (1, 1, 0),
        (2, 2, 1),
        (4, 4, 3),
        (8, 8, 7),
        (16, 16, 15),
        (7, 7, 6),
        (9, 9, 8),
    ],
)
def test_fit_dask_delayed_returns_an_uncomputed_graph_of_the_right_shape(n_blocks, n_local, n_merge):
    """The regression guard for the export's visualize bug, and the tree-shape contract at the same time.

    A graph, not a value: the return is a ``Delayed``, nothing was computed while building it (the deterministic
    tripwire would have caught a compute), and the task counts match the balanced pairwise tree exactly. ``7`` and
    ``9`` blocks are included because an odd level must carry its last node up unchanged rather than chain.
    """
    X = counting_array(n_blocks)
    COMPUTED_BLOCKS.clear()

    graph = MergeablePCA(n_components=3)._fit_dask_delayed(X)

    assert isinstance(graph, Delayed), "must return a graph so .visualize() can draw it"
    assert COMPUTED_BLOCKS == [], "graph construction computed a block"
    counts = count_tasks(graph)
    assert counts.get("local_pca", 0) == n_local, f"expected {n_local} local PCAs, got {counts}"
    assert counts.get("merge_pca", 0) == n_merge, f"expected {n_merge} merges, got {counts}"
    # The only task types of ours in the graph are the leaves and the merges; nothing else sneaks in. A single-block
    # input legitimately has NO merge task at all, so the expected set is derived from n_merge rather than hardcoded.
    ours = {"local_pca"} | ({"merge_pca"} if n_merge else set())
    assert set(counts) & {"local_pca", "merge_pca"} == ours


def test_fit_dask_delayed_graph_computes_to_the_fit_result():
    """``_fit_dask`` is exactly ``_fit_dask_delayed(...).compute()``, so the two paths cannot drift."""
    X = make_dask_array(n_samples=2048, n_features=40, rows=256)

    from_graph = MergeablePCA(n_components=7)._fit_dask_delayed(X).compute()
    from_fit = MergeablePCA(n_components=7).fit(X)

    assert isinstance(from_graph, PCASummary)
    assert from_graph.n_samples == from_fit._summary_.n_samples == 2048
    assert max_abs_error(from_graph.singular_values, from_fit._summary_.singular_values) < 1e-12
    assert max_abs_error(from_graph.components, from_fit._summary_.components) < 1e-10


def test_fit_dask_delayed_visualize_is_callable():
    """``.visualize()`` must exist and be callable: that is the whole point of the split.

    ``dot`` is not installed here, so the call is asserted to fail with the graphviz error rather than an
    AttributeError. Asserting "the attribute exists and is callable" is the portable half of the contract; a test that
    required a rendered file would fail on a machine without graphviz.
    """
    X = make_dask_array(n_samples=512, n_features=16, rows=256)
    graph = MergeablePCA(n_components=3)._fit_dask_delayed(X)

    assert hasattr(graph, "visualize") and callable(graph.visualize)
    try:
        graph.visualize()
    except Exception as error:  # noqa: BLE001 - graphviz absence is an environment fact, not an estimator bug
        assert "graphviz" in str(error).lower() or "dot" in str(error).lower(), f"unexpected error: {error!r}"


@pytest.mark.parametrize("batch_size", [25, 50, 100, 250])
def test_batch_size_splits_leaves_without_changing_the_result(batch_size):
    """``batch_size`` reshapes the tree only: merging is associative, so the answer is unchanged above roundoff."""
    X = make_dask_array(n_samples=1024, n_features=40, rows=256)
    reference = MergeablePCA(n_components=5).fit(X)

    pca = MergeablePCA(n_components=5, batch_size=batch_size).fit(X)

    # M2-adjacent: same retained spectrum.
    assert max_abs_error(pca.singular_values_, reference.singular_values_) < 1e-10
    assert max_abs_error(pca.mean_, reference.mean_) < 1e-12
    # M1 at k = 5 < d = 40.
    assert subspace_distance(pca.components_, reference.components_, k=5) < 1e-10


def test_batch_size_creates_more_leaves_than_row_chunks():
    """The leaf count is what ``batch_size`` actually controls, and it is observable in the graph."""
    X = make_dask_array(n_samples=1024, n_features=40, rows=256)
    row_chunks = len(X.chunks[0])

    graph = MergeablePCA(n_components=3, batch_size=128)._fit_dask_delayed(X)
    counts = count_tasks(graph)
    n_leaves = counts.get("local_pca", 0)

    assert n_leaves > row_chunks
    assert n_leaves == sum(-(-n // 128) for n in X.chunks[0])
    # And the tree still reduces to one root.
    assert counts.get("merge_pca", 0) == n_leaves - 1
    # Nothing was computed while building it.
    assert COMPUTED_BLOCKS == []


def test_batch_size_larger_than_the_block_adds_no_leaf():
    """A ``batch_size`` above the chunk size leaves the tree untouched: no padding leaves and no empty ones."""
    X = make_dask_array(n_samples=1024, n_features=40, rows=256)

    graph = MergeablePCA(n_components=3, batch_size=10_000).fit(X)._fit_dask_delayed(X)

    counts = count_tasks(graph)
    assert counts.get("local_pca", 0) == len(X.chunks[0])
    assert counts.get("merge_pca", 0) == len(X.chunks[0]) - 1


# =============================================================================
# transform / inverse_transform
# =============================================================================
def test_transform_projects_onto_the_components():
    """``transform`` is ``(X - mean_) @ components_.T`` and matches sklearn's projection up to sign (M1)."""
    data = make_data(1000, 24)
    reference = sklearn_pca(n_components=6).fit(data)

    pca = MergeablePCA(n_components=6).fit(da.from_array(data, chunks=(250, 24)))
    projected = np.asarray(pca.transform(da.from_array(data, chunks=(250, 24))))

    assert projected.shape == (1000, 6)
    # Signs are arbitrary per column, so compare |Z| column-aligned against the reference projection.
    sign = np.sign(np.einsum("ij,ij->j", projected, reference.transform(data)))
    sign[sign == 0] = 1.0
    assert max_abs_error(projected * sign, reference.transform(data)) < 1e-8


def test_whiten_gives_unit_sample_variance():
    """``whiten=True`` scales the public components so projected columns have unit sample variance (ddof=1)."""
    data = make_data(2000, 30)
    X = da.from_array(data, chunks=(250, 30))

    plain = MergeablePCA(n_components=8).fit(X)
    whitened = MergeablePCA(n_components=8, whiten=True).fit(X)

    projected = np.asarray(whitened.transform(X))
    assert np.allclose(np.var(projected, axis=0, ddof=1), 1.0, atol=1e-10), (
        f"whitened columns must have unit variance: {np.var(projected, axis=0, ddof=1)}"
    )
    # Without whiten the variance is the explained variance itself, not 1.
    plain_variance = np.var(np.asarray(plain.transform(X)), axis=0, ddof=1)
    assert not np.allclose(plain_variance, 1.0, atol=1e-3)
    # whiten must not touch the summary or the singular values.
    assert max_abs_error(whitened.singular_values_, plain.singular_values_) < 1e-15
    assert max_abs_error(whitened._summary_.singular_values, plain._summary_.singular_values) < 1e-15
    assert max_abs_error(whitened.components_, plain.components_) > 1e-6, "whiten must actually rescale"


def test_whiten_matches_sklearn_and_survives_a_constant_column():
    """Whitening agrees with sklearn, and a zero-variance direction must not become ``nan``."""
    data = make_data(1500, 20)
    constant_column = np.full((1500, 1), 7.0)
    X = da.from_array(np.hstack([data, constant_column]), chunks=(250, 21))
    reference = sklearn_pca(n_components=5, whiten=True).fit(np.hstack([data, constant_column]))

    pca = MergeablePCA(n_components=5, whiten=True).fit(X)

    assert np.all(np.isfinite(pca.components_)), "whiten must not produce nan"
    assert np.all(np.isfinite(pca.explained_variance_)), "a constant column must not produce nan/nan"
    assert max_abs_error(pca.explained_variance_, reference.explained_variance_) < 1e-10
    # The trailing zero-variance direction is left unrescaled rather than 0/0.
    assert np.all(np.isfinite(pca.components_[-1]))
    assert np.allclose(pca.explained_variance_ratio_, np.nan_to_num(reference.explained_variance_ratio_), atol=1e-10)


def test_inverse_transform_round_trips_the_retained_subspace():
    """``inverse_transform`` is exact on the RETAINED subspace, and cannot recover discarded directions.

    The correct claim is NOT "the round trip equals the data": with 15 of 20 components kept, the 5 discarded
    directions are gone by construction. What must hold is that reprojecting the reconstruction reproduces the original
    projection, and that the reconstruction is exactly the orthogonal projection of the centered data.
    """
    data = make_data(800, 20)
    X = da.from_array(data, chunks=(200, 20))
    pca = MergeablePCA(n_components=15).fit(X)

    projected = pca.transform(X)
    recovered = np.asarray(pca.inverse_transform(projected))

    assert recovered.shape == (800, 20)
    # Round trip on the retained subspace: reprojecting the reconstruction reproduces the original projection.
    reprojected = np.asarray(pca.transform(da.from_array(recovered, chunks=(200, 20))))
    assert max_abs_error(reprojected, np.asarray(projected)) < 1e-8
    # The reconstruction is the component of the centered data inside the retained subspace. Nothing else is claimed.
    # ``+ pca.mean_`` is REQUIRED: inverse_transform returns data in the ORIGINAL feature space (sklearn does the
    # same), so the reconstruction is mean + retained, not the retained part alone.
    centered = data - pca.mean_
    retained = pca.mean_ + (centered @ pca.components_.T) @ pca.components_
    assert max_abs_error(recovered, retained) < 1e-8


def test_fit_transform_returns_the_projection():
    """``fit_transform`` returns the projection (not the estimator), matching sklearn's return type."""
    data = make_data(400, 16)
    X = da.from_array(data, chunks=(100, 16))

    pca = MergeablePCA(n_components=5)
    projected = pca.fit_transform(X)

    assert isinstance(projected, da.Array)
    assert projected.shape == (400, 5)
    assert pca.n_components_ == 5, "fit_transform must also fit the estimator"


# =============================================================================
# Refusals
# =============================================================================
@pytest.mark.parametrize(
    ("kwargs", "expected_fragment"),
    [
        ({"n_components": 0}, "n_components must be None or a positive integer"),
        ({"n_components": -3}, "n_components must be None or a positive integer"),
        ({"n_components": 2.5}, "n_components must be None or a positive integer"),
        ({"n_components": "8"}, "n_components must be None or a positive integer"),
        ({"n_components": True}, "n_components must be None or a positive integer"),
        ({"local_rank": 0}, "local_rank must be None or a positive integer"),
        ({"local_rank": -1}, "local_rank must be None or a positive integer"),
        ({"local_rank": 3.5}, "local_rank must be None or a positive integer"),
        ({"batch_size": 0}, "batch_size must be None or a positive integer"),
        ({"batch_size": -10}, "batch_size must be None or a positive integer"),
        ({"batch_size": 1.5}, "batch_size must be None or a positive integer"),
        ({"whiten": 1}, "whiten must be a bool"),
        ({"whiten": "yes"}, "whiten must be a bool"),
        ({"whiten": None}, "whiten must be a bool"),
        ({"copy": 1}, "copy must be a bool"),
        ({"copy": "no"}, "copy must be a bool"),
    ],
)
def test_constructor_refuses_bad_values(kwargs, expected_fragment):
    """Constructor validation is immediate, not deferred to ``fit``, and names the offending parameter.

    ``bool`` is rejected for the integer parameters because ``isinstance(True, int)`` is ``True`` in Python; without
    that check ``n_components=True`` would silently mean "one component".
    """
    with pytest.raises(ValueError, match=expected_fragment):
        MergeablePCA(**kwargs)


def test_fit_refuses_1d_dask_input():
    """A 1-D array is refused before any indexing of ``chunks``, because one axis cannot be both kinds.

    Only the 1-D case remains here. The 3-D case this test used to assert alongside it is no longer a refusal: input
    of more than two dimensions is now supported through an explicit axis policy (``axis_names`` / ``feature_axes`` /
    ``sample_axes``), which the dedicated tests in ``test_mergeable_pca_axes.py`` cover, including the rule that the
    feature axes come LAST by default.
    """
    flat = da.from_array(make_data(10, 1).ravel(), chunks=5)

    with pytest.raises(ValueError, match="2-dimensional"):
        MergeablePCA(n_components=1).fit(flat)


def test_fit_refuses_features_split_across_chunks_and_names_the_remedy():
    """The chunking precondition is intrinsic: two summaries over different column sets cannot be stacked.

    The message must name ``rechunk({1: -1})`` and the remedy must ACTUALLY WORK, which is asserted by applying it.
    """
    X = da.from_array(make_data(200, 10), chunks=(50, 5))

    with pytest.raises(ValueError) as caught:
        MergeablePCA(n_components=2).fit(X)
    message = str(caught.value)
    assert "complete feature dimension in a single Dask chunk" in message
    assert "rechunk" in message, "the message must name the remedy"
    assert "{1: -1}" in message, "the message must name the exact rechunk call"

    # The named remedy is not decorative: it produces an array this estimator accepts.
    rechunked = X.rechunk({1: -1})
    assert len(rechunked.chunks[1]) == 1
    assert np.array_equal(np.asarray(rechunked), np.asarray(X))
    assert MergeablePCA(n_components=2).fit(rechunked).n_samples_ == 200


def test_fit_refuses_n_components_above_the_achievable_rank():
    """Requesting more components than ``min(n_samples, n_features)`` is refused, with both bounds named."""
    X = make_dask_array(n_samples=200, n_features=50, rows=100)

    with pytest.raises(ValueError) as caught:
        MergeablePCA(n_components=60).fit(X)
    message = str(caught.value)
    assert "n_components=60 exceeds" in message
    assert "min(n_samples=200, n_features=50)" in message
    assert "Reduce n_components" in message

    # Above n_samples too: the same refusal, now limited by the row count.
    Y = da.from_array(make_data(30, 50), chunks=(15, 50))
    with pytest.raises(ValueError, match="n_components=40 exceeds min"):
        MergeablePCA(n_components=40).fit(Y)


def test_fit_refuses_a_root_summary_too_small_for_n_components():
    """The one guard that makes ``local_rank < n_components`` safe: never silently return fewer components.

    A single block truncated to rank 2 can only reach rank 2, so asking for 5 components must fail loudly.
    """
    X = counting_array(n_blocks=1, rows=200, features=8)

    with pytest.raises(ValueError) as caught:
        MergeablePCA(n_components=5, local_rank=2).fit(X)
    message = str(caught.value)
    assert "root summary rank 2 < n_components=5" in message
    assert "local_rank" in message, "the message must name the remedy knob"
    assert "n_components" in message

    # The same combination is fine once the leaves can reach the requested rank.
    assert MergeablePCA(n_components=5, local_rank=8).fit(X).n_components_ == 5


def test_fit_refuses_zero_rows():
    """An empty array has no PCA; refuse rather than return an empty component set."""
    X = da.from_array(np.zeros((0, 5)), chunks=(1, 5))
    with pytest.raises(ValueError, match="zero rows"):
        MergeablePCA(n_components=1).fit(X)


def test_fit_refuses_zero_features():
    """Zero features is as undefined as zero rows, and is refused with its own message."""
    X = da.from_array(np.zeros((5, 0)), chunks=(5, 1))
    with pytest.raises(ValueError, match="zero features"):
        MergeablePCA(n_components=1).fit(X)


def test_methods_refuse_to_run_before_fit():
    """Every method that needs fitted attributes says so, and names the fix."""
    pca = MergeablePCA(n_components=3)

    with pytest.raises(ValueError, match="not fitted yet"):
        pca.transform(da.from_array(make_data(10, 4), chunks=(5, 4)))
    with pytest.raises(ValueError, match="not fitted yet"):
        pca.inverse_transform(da.from_array(make_data(10, 3), chunks=(5, 3)))


def test_transform_refuses_a_feature_count_mismatch():
    """Projecting data with a different number of features would be meaningless, so it is refused."""
    X = da.from_array(make_data(100, 10), chunks=(50, 10))
    pca = MergeablePCA(n_components=2).fit(X)

    with pytest.raises(ValueError, match="features but this estimator was fitted on 10"):
        pca.transform(da.from_array(make_data(100, 7), chunks=(50, 7)))


def test_transform_refuses_non_2d_input():
    """``transform`` validates its own input rather than failing inside a Dask matmul."""
    pca = MergeablePCA(n_components=2).fit(da.from_array(make_data(100, 10), chunks=(50, 10)))

    with pytest.raises(ValueError, match="2-dimensional"):
        pca.transform(da.from_array(make_data(10, 1).ravel(), chunks=5))


def test_inverse_transform_refuses_a_column_count_mismatch():
    """``inverse_transform`` needs exactly ``n_components_`` columns and names the count it kept."""
    X = da.from_array(make_data(100, 10), chunks=(50, 10))
    pca = MergeablePCA(n_components=3).fit(X)

    with pytest.raises(ValueError, match="kept 3 components"):
        pca.inverse_transform(da.from_array(make_data(100, 7), chunks=(50, 7)))


# =============================================================================
# What must stay ALLOWED
# =============================================================================
@pytest.mark.parametrize(
    ("n_samples", "n_features", "rows", "n_components"),
    [
        (100, 10, 50, 1),
        (100, 10, 25, 10),  # n_components == n_features, single feature chunk
        (100, 10, 100, 5),  # exactly one row chunk
        (64, 8, 8, 8),
    ],
)
def test_valid_two_dimensional_single_feature_chunk_input_is_allowed(n_samples, n_features, rows, n_components):
    """Assert what must KEEP working, so a later tightening cannot over-refuse silently.

    Valid 2-D input, features in one chunk, feasible ``n_components``, across several chunkings.
    """
    X = da.from_array(make_data(n_samples, n_features), chunks=(rows, n_features))

    pca = MergeablePCA(n_components=n_components).fit(X)

    assert pca.n_samples_ == n_samples
    assert pca.n_features_in_ == n_features
    assert pca.n_components_ == n_components


def test_constructor_stores_arguments_verbatim_and_fits_nothing():
    """All constructor arguments are stored verbatim and no work happens at construction."""
    pca = MergeablePCA()

    assert pca.n_components is None
    assert pca.whiten is False
    assert pca.copy is True
    assert pca.batch_size is None
    assert pca.local_rank is None
    assert not hasattr(pca, "_summary_"), "construction must not fit anything"


def test_fit_accepts_numpy_and_dask_input_alike():
    """Both documented input kinds are accepted and agree (M2-adjacent on the singular values)."""
    data = make_data(400, 16)

    from_numpy = MergeablePCA(n_components=4).fit(data)
    from_dask = MergeablePCA(n_components=4).fit(da.from_array(data, chunks=(100, 16)))

    assert max_abs_error(from_numpy.singular_values_, from_dask.singular_values_) < 1e-12
    assert from_numpy.n_samples_ == from_dask.n_samples_


def test_result_is_independent_of_the_scheduler():
    """The graph is a real Dask graph, so a non-default scheduler must give the same answer."""
    data = make_data(1024, 40)
    X = da.from_array(data, chunks=(256, 40))
    graph = MergeablePCA(n_components=5)._fit_dask_delayed(X)

    threaded = graph.compute(scheduler="threads", num_workers=4)
    synchronous = graph.compute(scheduler="synchronous")

    assert max_abs_error(threaded.singular_values, synchronous.singular_values) < 1e-12
    assert threaded.n_samples == synchronous.n_samples == 1024
