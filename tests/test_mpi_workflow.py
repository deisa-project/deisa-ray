from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

import pytest

from tests.utils import ray_cluster  # noqa: F401


@pytest.mark.skipif(
    shutil.which("mpirun") is None and shutil.which("mpiexec") is None,
    reason="MPI launcher is not installed",
)
def test_deisa_ray_with_mpi_comm(ray_cluster) -> None:  # noqa: F811
    pytest.importorskip("mpi4py")

    launcher = shutil.which("mpirun") or shutil.which("mpiexec")
    assert launcher is not None

    runner = pathlib.Path(__file__).with_name("mpi_full_workflow_runner.py")
    result = subprocess.run(
        [launcher, "-n", "4", sys.executable, str(runner)],
        cwd=pathlib.Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
