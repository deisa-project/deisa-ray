import pytest
import ray
import subprocess
import time
import shutil


def test_init_without_running_head_raises(monkeypatch):
    """Calling init() without a running Ray head should fail.

    This simulates a user not having executed `ray start --head` by ensuring
    no local Ray instance is initialized and no explicit address is set.
    """
    from doreisa.head_node import init

    # Ensure no Ray context is active and no implicit address is provided
    ray.shutdown()
    monkeypatch.delenv("RAY_ADDRESS", raising=False)

    # Expect an error when trying to connect to a non-existent cluster
    with pytest.raises(Exception):
        init()


def test_init_with_running_head_os_level(monkeypatch):
    """Starting a head via CLI should allow init() to connect successfully."""
    from doreisa.head_node import init

    # Skip if the ray CLI is unavailable (e.g., minimal CI envs)
    if shutil.which("ray") is None:
        pytest.skip("ray CLI is not available in PATH")

    # Ensure a clean state and no implicit addresses
    ray.shutdown()
    monkeypatch.delenv("RAY_ADDRESS", raising=False)

    # Ensure any stale Ray clusters are stopped to avoid port conflicts
    subprocess.run([
        "ray", "stop", "--force"
    ], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Start a Ray head at OS level
    subprocess.run(
        [
            "ray",
            "start",
            "--head",
            "--port=0",  # random open port to avoid collisions
            "--disable-usage-stats",
            "--num-cpus=1",
            "--dashboard-host",
            "127.0.0.1",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        # Give the head a brief moment to come up
        time.sleep(1.0)

        # Should connect to the running head via address="auto"
        init()
        assert ray.is_initialized()
    finally:
        ray.shutdown()
        subprocess.run(
            ["ray", "stop", "--force"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )


