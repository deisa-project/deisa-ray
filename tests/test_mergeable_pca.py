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
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# =============================================================================
"""
Tests for the mergeable PCA primitives (local summary, exact merge, balanced merge tree).

Why these tests do not compare components elementwise
-----------------------------------------------------
The sign of an eigenvector is arbitrary: LAPACK is free to return ``+v`` or ``-v``, and it does so depending on the
matrix's conditioning. A test asserting raw component equality therefore fails for a correct implementation and passes
for a broken one, which is the worst of both. Every accuracy assertion here is sign-invariant, via one of:

- **M1** ``subspace_distance(a, b, k)``: ``1 - min singular value`` of the overlap of the two leading ``k``-dimensional
  row subspaces. 0 means "same subspace", independent of signs and of any rotation inside the subspace. Both operands
  must be cut to the SAME ``k`` first, otherwise the truncated subspace is trivially a subspace of the full one.
- **M2** ``rel_var_error(sv_summary, sv_ref_full)``: relative error of the retained total variance against the
  FULL-rank batch reference. This is the metric for truncation: it falls monotonically as ``local_rank`` grows, which
  the raw per-component error does NOT do (it reads ~2.0 at exact reconstruction, i.e. maximally different).
- **M3** ``row_sign_fixed_error(a, b)``: per-row minimum of ``|a_i - b_i|`` and ``|a_i + b_i|``. Used only to pin the
  sign-invariance of the code, never as a quality metric.

Tolerances follow the technical spec: singular values and subspace distance ``< 1e-10``, means ``< 1e-12``, the
permutation spread of an associative merge ``< 1e-8``. All of those sit one to three orders of magnitude above the
residuals actually observed (1e-13 to 1e-16) because the exact digits are BLAS dependent.
"""

from __future__ import annotations

import gc
import itertools
import pickle

import numpy as np
import pytest

from deisa.ray.mergeable_pca import PCASummary, local_pca, merge_pca, merge_tree

# Number of random merge orders used by the associativity test. The spec requires at least 500.
N_PERMUTATIONS = 500


# ------------------------------------------------------------------------------ fixtures and references
# -----------------------------------------------------------------------------------------------------------
def make_data(n_samples: int, n_features: int, seed: int = 0) -> np.ndarray:
    """Reproducible data block; a fixed RandomState keeps every number in this file stable."""
    return np.random.RandomState(seed).randn(n_samples, n_features)


def batch_pca(X: np.ndarray) -> PCASummary:
    """
    Batch PCA reference on the full array: center everything, then one SVD.

    numpy only, because scipy and scikit-learn are not installed in this repo.
    """
    Xc = np.asarray(X, dtype=np.float64)
    mean = Xc.mean(axis=0)
    _, singular_values, components = np.linalg.svd(Xc - mean, full_matrices=False)
    return PCASummary(
        n_samples=int(Xc.shape[0]),
        mean=mean,
        components=components,
        singular_values=singular_values,
    )


def block_summaries(X: np.ndarray, n_blocks: int, rank: int | None = None) -> list[PCASummary]:
    """Split rows into ``n_blocks`` contiguous blocks and summarize each one."""
    return [local_pca(block, rank=rank) for block in np.array_split(X, n_blocks)]


# ------------------------------------------------------------------------------ sign-invariant metrics
# -----------------------------------------------------------------------------------------------------------
def subspace_distance(components_a: np.ndarray, components_b: np.ndarray, k: int) -> float:
    """
    M1. Distance between the leading ``k``-dimensional row subspaces of two component sets.

    ``1 - min singular value`` of the overlap of the two orthonormal bases. Range ``[0, 1]``, 0 = identical subspace.
    Sign invariant, and invariant to any orthogonal change of basis inside the retained subspace.

    ``k`` MUST be strictly smaller than the feature dimension, or the metric is vacuous: two ``d``-dimensional
    subspaces of ``R**d`` are the whole space, so they are always "identical" no matter how wrong either one is. Every
    call site below therefore picks ``k < d`` explicitly, and :func:`assert_subspace` rejects a vacuous ``k``.
    """
    if k >= components_a.shape[1]:
        raise AssertionError(f"subspace_distance needs k < n_features, got k={k}, n_features={components_a.shape[1]}")
    qa, _ = np.linalg.qr(components_a[:k].T)
    qb, _ = np.linalg.qr(components_b[:k].T)
    return float(1.0 - np.linalg.svd(qa.T @ qb, compute_uv=False).min())


def rel_var_error(singular_values: np.ndarray, singular_values_ref: np.ndarray) -> float:
    """
    M2. Relative error of the retained total variance, against the FULL-rank batch reference.

    ``singular_values_ref`` must be untruncated, otherwise this measures the truncation itself.
    """
    total_ref = np.sum(singular_values_ref**2)
    return float(abs(np.sum(singular_values**2) - total_ref) / total_ref)


def row_sign_fixed_error(components_a: np.ndarray, components_b: np.ndarray) -> float:
    """M3. Per-row error after allowing the best of the two signs. Sign invariant; not a quality metric."""
    k = min(components_a.shape[0], components_b.shape[0])
    return float(
        max(
            min(np.linalg.norm(components_a[i] - components_b[i]), np.linalg.norm(components_a[i] + components_b[i]))
            for i in range(k)
        )
    )


def max_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    """Max absolute elementwise error; only ever used on sign-free quantities (singular values, means)."""
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


# ------------------------------------------------------------------------------ local summary
# -----------------------------------------------------------------------------------------------------------
def test_local_pca_reproduces_batch_pca_on_one_block():
    """A single block summarized locally is the batch PCA of that block (full local rank)."""
    X = make_data(400, 12)
    reference = batch_pca(X)
    summary = local_pca(X)

    assert summary.n_samples == 400
    assert summary.components.shape == (12, 12)
    assert max_abs_error(summary.singular_values, reference.singular_values) < 1e-10
    assert max_abs_error(summary.mean, reference.mean) < 1e-12
    # M1: the principal subspace is the same one, whatever the signs. k = 6 < d = 12, so it is not vacuous.
    assert subspace_distance(summary.components, reference.components, k=6) < 1e-10


def test_local_pca_does_not_mutate_its_input():
    """The summary is built from a fresh centered copy, so the caller's block survives untouched."""
    X = make_data(50, 6)
    before = X.copy()
    local_pca(X)
    assert np.array_equal(X, before)


def test_local_pca_rank_truncation_keeps_leading_components():
    """``rank`` caps the local summary and keeps the leading directions, in order."""
    X = make_data(200, 10)
    full = local_pca(X)
    truncated = local_pca(X, rank=4)

    assert truncated.rank == 4
    assert max_abs_error(truncated.singular_values, full.singular_values[:4]) < 1e-10
    # M1: the retained subspace is the leading one of the full summary.
    assert subspace_distance(truncated.components, full.components, k=4) < 1e-10


def test_local_pca_casts_to_float64():
    """A non-float block is cast, so summaries are always float64 regardless of the input dtype."""
    summary = local_pca(np.arange(12, dtype=np.int32).reshape(4, 3))
    assert summary.mean.dtype == np.float64
    assert summary.components.dtype == np.float64
    assert summary.singular_values.dtype == np.float64


@pytest.mark.parametrize(
    ("block", "rank", "expected_fragment"),
    [
        (make_data(10, 3).ravel(), None, "2-dimensional"),
        (np.zeros((0, 4)), None, "zero rows"),
        (make_data(10, 3), 0, "positive integer"),
        (make_data(10, 3), 2.5, "positive integer"),
    ],
)
def test_local_pca_refuses_bad_input(block, rank, expected_fragment):
    """Refuse, do not approximate: every rejection names the offending condition."""
    with pytest.raises(ValueError, match=expected_fragment):
        local_pca(block, rank=rank)


# ------------------------------------------------------------------------------ exact merge
# -----------------------------------------------------------------------------------------------------------
def test_two_way_merge_reproduces_batch_pca():
    """The merge identity is exact: two summaries in, the batch PCA of the union out."""
    X = make_data(1000, 20)
    reference = batch_pca(X)
    merged = merge_pca(local_pca(X[:500]), local_pca(X[500:]))

    assert merged.n_samples == 1000
    # M2-adjacent magnitude check on the sign-free singular values.
    assert max_abs_error(merged.singular_values, reference.singular_values) < 1e-10
    assert max_abs_error(merged.mean, reference.mean) < 1e-12
    # M1: principal subspace at k = 10 < d = 20.
    assert subspace_distance(merged.components, reference.components, k=10) < 1e-10


def test_merge_is_exact_for_unevenly_sized_blocks():
    """Exactness does not depend on the blocks having the same number of rows."""
    X = make_data(600, 15)
    reference = batch_pca(X)
    parts = [local_pca(part) for part in np.split(X, [7, 7 + 133])]
    merged = merge_tree(parts)

    assert merged.n_samples == 600
    assert max_abs_error(merged.singular_values, reference.singular_values) < 1e-10
    assert max_abs_error(merged.mean, reference.mean) < 1e-12
    # M1 at k = 7 < d = 15.
    assert subspace_distance(merged.components, reference.components, k=7) < 1e-10


def test_merge_needs_no_source_samples():
    """
    A merge consumes summaries only.

    Proof: the source array is overwritten in place with NaN and dereferenced before the merge runs, and the merge
    still reproduces the batch PCA exactly. If any merge step consulted the samples, the result would be NaN.
    """
    X = make_data(800, 16)
    reference = batch_pca(X)
    summaries = block_summaries(X, 4)

    payload_elements = sum(
        s.mean.size + s.components.size + s.singular_values.size for s in summaries
    )  # the only data the merge will ever see
    raw_elements = X.size
    assert payload_elements < raw_elements, "the summary must be smaller than the samples it replaces"

    X[:] = np.nan  # destroy every sample in place
    del X
    gc.collect()

    merged = merge_tree(summaries)
    assert merged.n_samples == 800
    assert max_abs_error(merged.singular_values, reference.singular_values) < 1e-10
    assert max_abs_error(merged.mean, reference.mean) < 1e-12
    # M1 at k = 8 < d = 16: no NaN leaked in, the subspace is still the batch one.
    assert subspace_distance(merged.components, reference.components, k=8) < 1e-10


def test_merge_without_mean_correction_is_not_exact():
    """
    The between-block mean-correction term is what makes the merge exact.

    This test keeps the counterfactual in the suite on purpose: it builds the merge WITHOUT the Chan-Golub-LeVeque
    correction row and asserts the result is measurably wrong, while the real :func:`merge_pca` is exact. If someone
    "optimizes" the correction away, this test plus the exactness tests above fail together.
    """
    X = make_data(1000, 20)
    reference = batch_pca(X)
    a, b = local_pca(X[:500]), local_pca(X[500:])

    # The same merge with the correction row omitted.
    compact = np.vstack((a.singular_values[:, None] * a.components, b.singular_values[:, None] * b.components))
    _, singular_values, components = np.linalg.svd(compact, full_matrices=False)

    assert max_abs_error(singular_values, reference.singular_values) > 1e-6
    # M1 at k = 10 < d = 20: the subspaces differ measurably, not at roundoff level.
    assert subspace_distance(components, reference.components, k=10) > 1e-6
    # And the implemented merge, which keeps the correction, is exact.
    assert max_abs_error(merge_pca(a, b).singular_values, reference.singular_values) < 1e-10


def test_merge_refuses_mismatched_feature_dimensions():
    """Two summaries over different feature bases cannot be stacked; the error names the remedy."""
    a = local_pca(make_data(20, 4))
    b = local_pca(make_data(20, 6))
    with pytest.raises(ValueError, match="different feature dimensions"):
        merge_pca(a, b)


def test_merge_refuses_empty_sample_count():
    """A summary with n_samples < 1 is refused rather than silently merged."""
    a = local_pca(make_data(20, 4))
    empty = PCASummary(n_samples=0, mean=a.mean, components=a.components, singular_values=a.singular_values)
    with pytest.raises(ValueError, match="at least one sample"):
        merge_pca(a, empty)


# ------------------------------------------------------------------------------ balanced merge tree
# -----------------------------------------------------------------------------------------------------------
def test_tree_merge_reproduces_batch_pca():
    """A balanced tree over 16 blocks reproduces the batch PCA of the whole array."""
    X = make_data(4096, 50)
    reference = batch_pca(X)
    merged = merge_tree(block_summaries(X, 16))

    assert merged.n_samples == 4096
    assert merged.rank == 50
    assert max_abs_error(merged.singular_values, reference.singular_values) < 1e-10
    assert max_abs_error(merged.mean, reference.mean) < 1e-12
    # M1 at k = 25 < d = 50.
    assert subspace_distance(merged.components, reference.components, k=25) < 1e-10


def test_tree_merge_reproduces_batch_pca_over_eight_blocks():
    """The same identity at a deeper-than-two level of blocks, with 8 blocks."""
    X = make_data(2048, 32)
    reference = batch_pca(X)
    merged = merge_tree(block_summaries(X, 8))

    assert merged.n_samples == 2048
    assert max_abs_error(merged.singular_values, reference.singular_values) < 1e-10
    assert max_abs_error(merged.mean, reference.mean) < 1e-12
    # M1 at k = 16 < d = 32.
    assert subspace_distance(merged.components, reference.components, k=16) < 1e-10


@pytest.mark.parametrize("n_blocks", [2, 4, 8, 16, 32])
def test_root_rank_saturates_at_feature_dim(n_blocks):
    """
    With full local rank the summary rank saturates at the feature dimension and stops there.

    This is the docstring's scalability claim, so it is asserted rather than described: no block count makes the root
    rank exceed ``d``. The singular-value check rides along because a saturated rank that is NOT exact would mean the
    saturation lost information.
    """
    X = make_data(4096, 50)
    reference = batch_pca(X)
    merged = merge_tree(block_summaries(X, n_blocks))

    assert merged.rank == 50, f"root rank grew with the tree for {n_blocks} blocks"
    assert max_abs_error(merged.singular_values, reference.singular_values) < 1e-10


@pytest.mark.parametrize("n_blocks", [7, 9])
def test_merge_tree_handles_odd_block_counts(n_blocks):
    """An odd level carries its last summary up unchanged, so an odd block count stays exact."""
    X = make_data(1024, 24)
    reference = batch_pca(X)
    merged = merge_tree(block_summaries(X, n_blocks))

    assert merged.n_samples == 1024
    assert max_abs_error(merged.singular_values, reference.singular_values) < 1e-10
    assert max_abs_error(merged.mean, reference.mean) < 1e-12
    # M1 at k = 12 < d = 24.
    assert subspace_distance(merged.components, reference.components, k=12) < 1e-10


def test_merge_tree_single_summary_is_the_base_case():
    """One summary in, that same summary out: no SVD, no merge."""
    X = make_data(64, 8)
    summary = local_pca(X)
    merged = merge_tree([summary])

    assert merged is summary
    assert merged.n_samples == 64


def test_merge_tree_of_two_is_the_plain_merge():
    """Two summaries go through the pairwise path, which is the merge itself."""
    X = make_data(120, 10)
    summaries = block_summaries(X, 2)
    left, right = summaries
    merged = merge_tree(summaries)

    assert merged.n_samples == left.n_samples + right.n_samples
    # M2-adjacent: same total variance as the direct two-way merge of the same two leaves.
    assert rel_var_error(merged.singular_values, merge_pca(left, right).singular_values) < 1e-10
    assert max_abs_error(merged.singular_values, merge_pca(left, right).singular_values) < 1e-10


def test_merge_tree_refuses_no_summaries():
    """An empty reduction is undefined and is refused, not faked."""
    with pytest.raises(ValueError, match="no summaries to merge"):
        merge_tree([])


# ------------------------------------------------------------------------------ associativity
# -----------------------------------------------------------------------------------------------------------
def test_merge_is_associative():
    """
    ``merge(merge(a, b), c)`` and ``merge(a, merge(b, c))`` agree, and so does every other grouping.

    Associativity is the property that justifies the balanced tree in the first place: if it did not hold, the choice
    of merge order would change the answer. Tested both on the two groupings explicitly and on the spread over random
    permutations of 8 blocks.
    """
    X = make_data(1024, 30)
    reference = batch_pca(X)
    summaries = block_summaries(X, 8)
    a, b, c = summaries[0], summaries[1], summaries[2]

    left_grouped = merge_pca(merge_pca(a, b), c)
    right_grouped = merge_pca(a, merge_pca(b, c))

    assert left_grouped.n_samples == right_grouped.n_samples == a.n_samples + b.n_samples + c.n_samples
    # Sign-free quantities can be compared directly.
    assert max_abs_error(left_grouped.singular_values, right_grouped.singular_values) < 1e-8
    assert max_abs_error(left_grouped.mean, right_grouped.mean) < 1e-12
    # M1 at k = 15 < d = 30: the two groupings span the same principal subspace.
    assert subspace_distance(left_grouped.components, right_grouped.components, k=15) < 1e-8
    # M2-adjacent: same retained variance.
    assert rel_var_error(left_grouped.singular_values, right_grouped.singular_values) < 1e-8

    rng = np.random.RandomState(1)
    orders = list(itertools.permutations(range(8)))
    chosen = rng.choice(len(orders), size=N_PERMUTATIONS, replace=False)
    merged_singular_values = np.array([merge_tree([summaries[i] for i in orders[j]]).singular_values for j in chosen])

    # Per-component spread across merge orders, and deviation from the batch reference.
    assert float(np.max(merged_singular_values.max(axis=0) - merged_singular_values.min(axis=0))) < 1e-8
    assert max_abs_error(merged_singular_values, reference.singular_values) < 1e-8


# ------------------------------------------------------------------------------ local_rank truncation
# -----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("local_rank", [None, 50, 60, 100])
def test_local_rank_at_or_above_the_feature_dim_is_exact(local_rank):
    """``local_rank >= d`` truncates nothing: the merged summary is the batch PCA, root rank included."""
    X = make_data(4096, 50)
    reference = batch_pca(X)
    merged = merge_tree(block_summaries(X, 16, rank=local_rank))

    assert merged.rank == 50
    assert max_abs_error(merged.singular_values, reference.singular_values) < 1e-10
    assert max_abs_error(merged.mean, reference.mean) < 1e-12
    # M1 at k = 25 < d = 50: the subspace is the batch one even though the leaves were capped.
    assert subspace_distance(merged.components, reference.components, k=25) < 1e-10


@pytest.mark.parametrize("local_rank", [5, 10, 20, 40])
def test_local_rank_below_the_feature_dim_costs_variance(local_rank):
    """
    ``local_rank < d`` discards variance the merge cannot recover -- root rank saturation is NOT exactness.

    This is the honest counterpart of the test above and the guard against a tempting wrong claim: the root rank still
    saturates at ``d``, which can look like the truncation was free, but the retained directions are measurably wrong.
    Asserted with the two sign-invariant metrics, both of which must be far from the exact values.
    """
    X = make_data(4096, 50)
    reference = batch_pca(X)
    merged = merge_tree(block_summaries(X, 16, rank=local_rank))

    assert merged.rank == 50, "rank saturates at d regardless of the local rank"
    assert max_abs_error(merged.singular_values, reference.singular_values) > 1e-3
    # M2: measurable variance loss.
    assert rel_var_error(merged.singular_values, reference.singular_values) > 1e-3
    # M1 at k = 25 < d = 50: measurable subspace loss.
    assert subspace_distance(merged.components, reference.components, k=25) > 1e-6


def test_truncated_local_rank_degrades_monotonically():
    """
    As ``local_rank`` shrinks, both honest metrics degrade monotonically; full rank is exact.

    Monotonicity is the assertion, not a specific accuracy: the truncated rank has no universal "safe" value, but it
    always costs variance and always costs subspace accuracy in the same direction. This is precisely the test a naive
    per-component error cannot support: that metric is non-monotonic and reads ~2.0 (maximally different) at exact
    reconstruction, so a "lower is better" assertion on it is meaningless.

    The per-step slack absorbs float noise when two consecutive ranks produce the same summary; it is many orders of
    magnitude below the smallest measured step, so it cannot hide a real regression.
    """
    n_components = 10
    X = make_data(4096, 50)
    reference = batch_pca(X)

    ranks = (5, 10, 20, 40, None)
    subspace_distances = []
    variance_errors = []
    for local_rank in ranks:
        merged = merge_tree(block_summaries(X, 16, rank=local_rank))
        # M2: variance retained, against the FULL-rank batch reference.
        variance_errors.append(rel_var_error(merged.singular_values, reference.singular_values))
        # M1: subspace, both sides cut to the same k.
        subspace_distances.append(subspace_distance(merged.components, reference.components, k=n_components))

    slack = 1e-12
    for previous, current in zip(subspace_distances, subspace_distances[1:]):
        assert current <= previous + slack, f"M1 not monotone at ranks={ranks}: {subspace_distances}"
    for previous, current in zip(variance_errors, variance_errors[1:]):
        assert current <= previous + slack, f"M2 not monotone at ranks={ranks}: {variance_errors}"

    # Full local rank is exact on both metrics.
    assert variance_errors[-1] < 1e-10
    assert subspace_distances[-1] < 1e-10
    # And truncation is not free: the smallest rank is measurably worse than the largest truncated rank.
    assert variance_errors[0] > variance_errors[-2]


# ------------------------------------------------------------------------------ sign invariance of the metrics
# -----------------------------------------------------------------------------------------------------------
def test_sign_invariance_of_the_metrics():
    """
    M1 and M3 are sign invariant; the banned raw error is not. This documents why raw comparison is banned.

    Flipping the sign of one component row is the same subspace, so M1 and M3 must both read 0, while the raw signed
    error reads 2.0 (the maximum possible distance, ``|v - (-v)| = 2`` for unit rows).
    """
    X = make_data(512, 16)
    summary = local_pca(X)
    flipped = summary.components.copy()
    flipped[0] *= -1.0

    assert subspace_distance(summary.components, flipped, k=8) < 1e-12
    assert row_sign_fixed_error(summary.components, flipped) < 1e-12
    assert max_abs_error(summary.components[0], flipped[0]) > 1.0


# ------------------------------------------------------------------------------ picklability
# -----------------------------------------------------------------------------------------------------------
def test_primitives_and_summaries_are_picklable():
    """
    Everything here crosses a process boundary (the future MPI bridge ``branch_func``, ``dask.delayed``).

    Only module-level functions and plain dataclasses survive that, so this is a real contract, not a formality: a
    lambda or a closure introduced here would fail at ``mpirun`` runtime with "Can't pickle".
    """
    X = make_data(64, 8)
    summaries = block_summaries(X, 4)

    assert pickle.loads(pickle.dumps(local_pca)) is local_pca
    assert pickle.loads(pickle.dumps(merge_pca)) is merge_pca
    assert pickle.loads(pickle.dumps(merge_tree)) is merge_tree

    restored = pickle.loads(pickle.dumps(merge_tree(summaries)))
    assert restored.n_samples == merge_tree(summaries).n_samples
    assert max_abs_error(restored.singular_values, merge_tree(summaries).singular_values) == 0.0
