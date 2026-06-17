from __future__ import annotations

import os
import pathlib
import shlex
import shutil
import subprocess
import sys

import pytest

from tests.utils import ray_multinode_cluster  # noqa: F401


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
        command.extend(
            [
                "-x",
                "LD_LIBRARY_PATH",
                "-x",
                "MPI4PY_LIBMPI",
                "-x",
                "DEISA_RAY_ADDRESS",
                "-x",
                "PYTHONUNBUFFERED",
                "-x",
                "PYTHONFAULTHANDLER",
            ]
        )

    command.extend(["-n", "4", sys.executable, str(runner)])
    return command


def _launcher_lib_env(launcher: str, env: dict[str, str]) -> dict[str, str]:
    lib_dir = pathlib.Path(launcher).resolve().parents[1] / "lib"
    if not lib_dir.exists():
        return {}

    updates = {}

    libmpi = lib_dir / "libmpi.so"
    if libmpi.exists():
        updates["MPI4PY_LIBMPI"] = str(libmpi)

    current = env.get("LD_LIBRARY_PATH")
    updates["LD_LIBRARY_PATH"] = str(lib_dir) if not current else f"{lib_dir}{os.pathsep}{current}"
    return updates


def _prepend_launcher_lib_path(env: dict[str, str], launcher: str) -> None:
    env.update(_launcher_lib_env(launcher, env))


@pytest.fixture
def mpi_launcher_env(monkeypatch):
    launcher = shutil.which("mpirun") or shutil.which("mpiexec")
    if launcher is None:
        return

    for name, value in _launcher_lib_env(launcher, os.environ).items():
        monkeypatch.setenv(name, value)


@pytest.fixture
def mpi_ray_multinode_cluster(mpi_launcher_env, ray_multinode_cluster):
    return ray_multinode_cluster


@pytest.mark.skipif(
    shutil.which("mpirun") is None and shutil.which("mpiexec") is None,
    reason="MPI launcher is not installed",
)
def test_deisa_ray_with_mpi_comm(mpi_ray_multinode_cluster) -> None:  # noqa: F811
    pytest.importorskip("mpi4py")

    launcher = shutil.which("mpirun") or shutil.which("mpiexec")
    assert launcher is not None

    runner = pathlib.Path(__file__).with_name("mpi_full_workflow_runner.py")
    command = _build_mpi_command(launcher, runner)
    cwd = pathlib.Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    _prepend_launcher_lib_path(env, launcher)
    env["DEISA_RAY_ADDRESS"] = mpi_ray_multinode_cluster["cluster"].address
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONFAULTHANDLER"] = "1"
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
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
            f"DEISA_RAY_ADDRESS: {env['DEISA_RAY_ADDRESS']}",
            f"LD_LIBRARY_PATH: {env.get('LD_LIBRARY_PATH', '<unset>')}",
            f"MPI4PY_LIBMPI: {env.get('MPI4PY_LIBMPI', '<unset>')}",
            f"stdout:\n{stdout}",
            f"stderr:\n{stderr}",
        ]
    )

    assert result.returncode == 0, diagnostic
