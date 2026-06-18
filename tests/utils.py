import socket
import time

import numpy as np
import pytest
import ray
from ray.cluster_utils import Cluster


def pick_free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def start_ray_multinode_cluster(
    *,
    head_node_gcs_server_port: int,
) -> Cluster:
    """Start a local multinode Ray test cluster with isolated head ports."""
    cluster = Cluster(
        initialize_head=True,
        connect=False,
        head_node_args={
            "num_cpus": 1,
            "gcs_server_port": head_node_gcs_server_port,
            "dashboard_port": pick_free_port(),
        },
    )
    cluster.add_node(num_cpus=1)
    cluster.add_node(num_cpus=1)

    return cluster


@pytest.fixture
def ray_multinode_cluster(monkeypatch):
    cluster = start_ray_multinode_cluster(
        head_node_gcs_server_port=pick_free_port(),
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

    yield {
        "cluster": cluster,
    }

    ray.shutdown()
    cluster.shutdown()


def wait_for_head_node() -> None:
    """Wait until the head node is ready"""
    while True:
        try:
            a = ray.get_actor("simulation_head", namespace="deisa_ray")
            ray.get(a.ready.remote())
            return
        except ValueError:
            time.sleep(0.1)