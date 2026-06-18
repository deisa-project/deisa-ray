# TODO fix this with incoming PR on memory handling!
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import ray
from ray.cluster_utils import Cluster

from deisa.ray.types import DeisaArray
from tests.utils import wait_for_head_node, pick_free_port  # noqa: F401

NB_ITERATIONS = 240  # Should be enough to saturate the memory in case the chunks are not released
OBJECT_STORE_MEMORY = 80 * 1024 * 1024
SPILL_MONITOR_INTERVAL_S = 0.01


def _spilled_bytes(spilling_path: Path) -> int:
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(spilling_path):
        for filename in filenames:
            filepath = Path(dirpath) / filename
            try:
                if filepath.is_file():
                    total_size += filepath.stat().st_size
            except FileNotFoundError:
                pass

    return total_size


class SpillMonitor:
    def __init__(self, spilling_path: str) -> None:
        self.spilling_path = Path(spilling_path)
        self.max_spilled_bytes = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._poll_spilling_path)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop_event.set()
        self._thread.join()
        self.max_spilled_bytes = max(self.max_spilled_bytes, _spilled_bytes(self.spilling_path))

    def _poll_spilling_path(self) -> None:
        while not self._stop_event.is_set():
            self.max_spilled_bytes = max(self.max_spilled_bytes, _spilled_bytes(self.spilling_path))
            time.sleep(SPILL_MONITOR_INTERVAL_S)


@ray.remote(max_retries=0)
def head_script(enable_distributed_scheduling) -> None:
    """The head node checks that the values are correct"""
    from deisa.ray.window_handler import Deisa

    os.environ["DEISA_DISTRIBUTED_SCHEDULING"] = "1" if enable_distributed_scheduling else "0"

    d = Deisa()

    @d.register("array")
    def simulation_callback(array: list[DeisaArray]):
        pass
    d.execute_callbacks()


@ray.remote(num_cpus=0, max_retries=0)
def bridge_script(*, rank: int, port: int) -> None:
    from deisa.ray.bridge import Bridge
    from tests.comm_utils import NoOpComm
    arrays_md = {
        "array": {
            "global_shape": (1024, 1024),
            "chunk_shape": (1024, 1024),
            "chunk_position": (0, 0),
        }
    }

    bridge = Bridge(arrays_metadata=arrays_md, comm=NoOpComm())

    array = (rank + 1) * np.ones((1024, 1024), dtype=np.int32)
    for timestep in range(NB_ITERATIONS):
        bridge.send(array_name="array", chunk=timestep * array, timestep=timestep)

    bridge.close(timestep=NB_ITERATIONS)


@pytest.fixture
def ray_spilling_cluster(monkeypatch, tmp_path):
    spilling_path = tmp_path / "ray_spilling"
    ray_tmp_path = tempfile.mkdtemp(prefix="deisa_ray_")
    cluster = None
    try:
        cluster = Cluster(
            initialize_head=True,
            connect=False,
            head_node_args={
                "num_cpus": 1,
                "object_store_memory": OBJECT_STORE_MEMORY,
                "object_spilling_directory": str(spilling_path),
                "temp_dir": ray_tmp_path,
                "gcs_server_port": pick_free_port(),
                "dashboard_port": pick_free_port(),
            },
        )

        monkeypatch.setenv("DEISA_RAY_ADDRESS", cluster.address)
        monkeypatch.setenv("RAY_ADDRESS", cluster.address)

        ray.init(
            address=cluster.address,
            include_dashboard=False,
            log_to_driver=True,
            ignore_reinit_error=True,
            runtime_env={
                "env_vars": {
                    "DEISA_RAY_ADDRESS": cluster.address,
                    "RAY_ADDRESS": cluster.address,
                }
            },
        )

        yield str(spilling_path)
    finally:
        ray.shutdown()
        if cluster is not None:
            cluster.shutdown()
        shutil.rmtree(ray_tmp_path, ignore_errors=True)


@pytest.mark.parametrize("enable_distributed_scheduling", [True, False])
def test_memory_release(enable_distributed_scheduling, ray_spilling_cluster: str) -> None:  # noqa: F811
    """
    Perform a long simulation with a small object store spilling to disk. If the
    memory is not released correctly, the test will detect spilled objects on disk and
    fail.
    """
    with SpillMonitor(ray_spilling_cluster) as spill_monitor:
        head_ref = head_script.remote(enable_distributed_scheduling)
        wait_for_head_node()
        port = pick_free_port()

        worker = bridge_script.remote(
            rank=0,
            port=port,
        )

        ray.get([head_ref, worker])

    assert spill_monitor.max_spilled_bytes == 0
