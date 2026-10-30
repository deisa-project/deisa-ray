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
Tests for the MergeablePCA public API surface: ``from deisa.ray import MergeablePCA``.

These are packaging tests, not algorithm tests: the algebra is covered in test_mergeable_pca.py,
test_mergeable_pca_axes.py and test_mergeable_pca_estimator.py. What is checked here is that the estimator is
reachable from the top-level package, that it is advertised in ``__all__``, and that the re-export is the very same
object as the definition, so the two cannot drift apart silently.

Why a real import and not a source scan
---------------------------------------
A grep over ``__init__.py`` would pass on a file that re-exports a name that does not exist, or on an export guarded
so that it is skipped at import time. The only way to know the documented public import actually works is to execute
it, so every test below imports the name at run time, and ``test_import_works_in_a_clean_interpreter`` executes it in
a separate interpreter with no test-time state at all.
"""

from __future__ import annotations

import os
import subprocess
import sys

import deisa.ray
from deisa.ray.mergeable_pca import MergeablePCA as DefinedMergeablePCA

# The public methods the estimator contract promises. Used as a cheap "this is the estimator class" smoke check: the
# package could re-export any object with the right name, so the check is on the shape of the API, not the name.
EXPECTED_PUBLIC_METHODS = ("fit", "fit_transform", "transform", "inverse_transform")


def test_public_import_succeeds():
    """The documented import works: the name is bound by ``import deisa.ray``, not by a scan of the source."""
    from deisa.ray import MergeablePCA

    assert MergeablePCA is not None
    assert isinstance(MergeablePCA, type)


def test_mergeable_pca_is_exported_in_dunder_all():
    """``__all__`` advertises the name, so ``from deisa.ray import *`` and the docs agree with the import."""
    assert "MergeablePCA" in deisa.ray.__all__


def test_reexport_is_the_same_object_as_the_definition():
    """The re-export IS the definition, so renaming or re-implementing the class cannot silently leave a stale alias."""
    from deisa.ray import MergeablePCA

    assert MergeablePCA is DefinedMergeablePCA
    assert MergeablePCA is deisa.ray.mergeable_pca.MergeablePCA


def test_reexport_is_the_estimator_class():
    """Light smoke check: the exported object exposes the estimator methods and can be instantiated."""
    from deisa.ray import MergeablePCA

    for method in EXPECTED_PUBLIC_METHODS:
        assert callable(getattr(MergeablePCA, method)), f"MergeablePCA.{method} is missing or not callable"

    estimator = MergeablePCA(n_components=2)
    assert estimator.n_components == 2


def test_import_works_in_a_clean_interpreter():
    """The import does not depend on this test session: it works in a bare interpreter given only the package path."""
    source = "from deisa.ray import MergeablePCA; print(MergeablePCA.__name__)"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)

    completed = subprocess.run(
        [sys.executable, "-c", source],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert completed.returncode == 0, f"clean-interpreter import failed:\n{completed.stderr}"
    assert completed.stdout.strip() == "MergeablePCA", completed.stdout
