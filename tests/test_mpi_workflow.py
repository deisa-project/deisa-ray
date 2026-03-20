from __future__ import annotations

import pathlib
import shlex
import shutil
import subprocess
import sys

import pytest

from tests.utils import ray_cluster  # noqa: F401


def _build_mpi_command(launcher: str, runner: pathlib.Path) -> list[str]:
    """
    Build the MPI launcher command for the workflow runner.

    Parameters
    ----------
    launcher : str
        Absolute path to the MPI launcher executable.
    runner : pathlib.Path
        Path to the Python workflow runner script.

    Returns
    -------
    list[str]
        Command line used to launch the MPI workflow test.

    Notes
    -----
    OpenMPI may reject `-n 4` on constrained CI runners unless
    ``--oversubscribe`` is passed explicitly.
    """
    command = [launcher]

    if pathlib.Path(launcher).name in {"mpirun", "mpiexec"}:
        command.append("--oversubscribe")

    command.extend(["-n", "4", sys.executable, str(runner)])
    return command


@pytest.mark.skipif(
    shutil.which("mpirun") is None and shutil.which("mpiexec") is None,
    reason="MPI launcher is not installed",
)
def test_deisa_ray_with_mpi_comm(ray_cluster) -> None:  # noqa: F811
    pytest.importorskip("mpi4py")

    launcher = shutil.which("mpirun") or shutil.which("mpiexec")
    assert launcher is not None

    runner = pathlib.Path(__file__).with_name("mpi_full_workflow_runner.py")
    command = _build_mpi_command(launcher, runner)
    cwd = pathlib.Path(__file__).resolve().parents[1]
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )

    stdout = result.stdout.strip() or "<empty>"
    stderr = result.stderr.strip() or "<empty>"
    diagnostic = "\n".join(
        [
            "MPI workflow command failed",
            f"command: {shlex.join(command)}",
            f"cwd: {cwd}",
            f"returncode: {result.returncode}",
            f"stdout:\n{stdout}",
            f"stderr:\n{stderr}",
        ]
    )

    assert result.returncode == 0, diagnostic
