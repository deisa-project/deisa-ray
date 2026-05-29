import os
import socket
import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pytest
import ray
from ray.exceptions import GetTimeoutError


DEISA_NAMESPACE = "deisa_ray"
HEAD_ACTOR_NAME = "simulation_head"
DEFAULT_RAY_TIMEOUT_S = 45.0


def pick_free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture()
def ray_cluster():
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    previous_ray_address = os.environ.get("RAY_ADDRESS")
    if not ray.is_initialized():
        os.environ.pop("RAY_ADDRESS", None)
        ray.init(log_to_driver=False)
    publish_current_ray_address()
    cleanup_deisa_actors()
    try:
        yield ray.get_runtime_context().gcs_address
    finally:
        cleanup_deisa_actors()
        if previous_ray_address is None:
            os.environ.pop("RAY_ADDRESS", None)
        else:
            os.environ["RAY_ADDRESS"] = previous_ray_address


def publish_current_ray_address() -> None:
    os.environ["RAY_ADDRESS"] = ray.get_runtime_context().gcs_address


def cleanup_deisa_actors(timeout_s: float = 10.0) -> None:
    """Kill detached test actors so a reused Ray runtime starts each test cleanly."""
    if not ray.is_initialized():
        return

    actor_names = _live_deisa_actor_names()
    for name in actor_names:
        try:
            ray.kill(ray.get_actor(name, namespace=DEISA_NAMESPACE), no_restart=True)
        except (ValueError, RuntimeError):
            pass

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _live_deisa_actor_names():
            return
        time.sleep(0.05)


def _live_deisa_actor_names() -> set[str]:
    actors = ray.util.list_named_actors(all_namespaces=True)
    return {
        actor["name"]
        for actor in actors
        if actor.get("namespace") == DEISA_NAMESPACE and actor.get("name")
    }


def wait_for_head_node(timeout_s: float = DEFAULT_RAY_TIMEOUT_S) -> None:
    """Wait until the head node is ready"""
    deadline = time.monotonic() + timeout_s
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            a = ray.get_actor(HEAD_ACTOR_NAME, namespace=DEISA_NAMESPACE)
            ray.get(a.ready.remote(), timeout=1)
            return
        except (ValueError, GetTimeoutError) as exc:
            last_error = exc
            time.sleep(0.05)

    raise TimeoutError(f"{DEISA_NAMESPACE}/{HEAD_ACTOR_NAME} was not ready within {timeout_s}s") from last_error


@dataclass(frozen=True)
class WorkerSpec:
    rank: int
    position: tuple[int, ...]
    chunks_per_dim: tuple[int, ...]
    chunk_size: tuple[int, ...]
    nb_iterations: int
    nb_nodes: int
    node_id: str | None = None
    array_name: str | list[str] = "array"
    dtype: np.dtype = np.int32  # type: ignore[type-arg]
    start_iteration: int = 0
    sleep_before_send: float = 0
    sleep_between_sends: float = 0


class RayWorkflowHarness:
    def __init__(self, *, timeout_s: float = DEFAULT_RAY_TIMEOUT_S) -> None:
        self.timeout_s = timeout_s
        self.port = pick_free_port()
        self._refs: list[tuple[str, ray.ObjectRef]] = []

    def start_head(self, remote_fn: Any, *args: Any, label: str = "head", **kwargs: Any) -> ray.ObjectRef:
        ref = remote_fn.remote(*args, **kwargs)
        self._refs.append((label, ref))
        wait_for_head_node(timeout_s=self.timeout_s)
        return ref

    def start_simple_workers(
        self,
        specs: Iterable[WorkerSpec],
        *,
        worker: Any = None,
        label_prefix: str = "worker",
    ) -> list[ray.ObjectRef]:
        worker = worker or simple_worker
        refs = []
        for spec in specs:
            ref = worker.remote(
                rank=spec.rank,
                position=spec.position,
                chunks_per_dim=spec.chunks_per_dim,
                chunk_size=spec.chunk_size,
                nb_iterations=spec.nb_iterations,
                nb_nodes=spec.nb_nodes,
                port=self.port,
                node_id=spec.node_id,
                array_name=spec.array_name,
                dtype=spec.dtype,
                start_iteration=spec.start_iteration,
                _sleep_b4_send=spec.sleep_before_send,
                _sleep_intra_send=spec.sleep_between_sends,
            )
            refs.append(ref)
            self._refs.append((f"{label_prefix}-{spec.rank}", ref))
        return refs

    def wait(self, refs: Iterable[ray.ObjectRef] | None = None) -> list[Any]:
        labeled_refs = self._refs if refs is None else [(f"ref-{i}", ref) for i, ref in enumerate(refs)]
        return get_all_with_timeout(labeled_refs, timeout_s=self.timeout_s)

    def assert_scheduling_actor_count(self, expected: int) -> None:
        simulation_head = ray.get_actor(HEAD_ACTOR_NAME, namespace=DEISA_NAMESPACE)
        actors = ray.get(simulation_head.list_scheduling_actors.remote(), timeout=self.timeout_s)
        assert len(actors) == expected


@pytest.fixture()
def ray_workflow(ray_cluster) -> RayWorkflowHarness:  # noqa: F811
    return RayWorkflowHarness()


def get_all_with_timeout(
    refs: Iterable[tuple[str, ray.ObjectRef]] | Iterable[ray.ObjectRef],
    *,
    timeout_s: float = DEFAULT_RAY_TIMEOUT_S,
) -> list[Any]:
    labeled_refs = _normalize_labeled_refs(refs)
    pending = {ref: label for label, ref in labeled_refs}
    results: dict[ray.ObjectRef, Any] = {}
    deadline = time.monotonic() + timeout_s

    while pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            labels = ", ".join(pending.values())
            raise TimeoutError(f"Timed out after {timeout_s}s waiting for Ray refs: {labels}")

        ready, still_pending = ray.wait(list(pending), num_returns=1, timeout=min(1, remaining))
        if not ready:
            pending = {ref: pending[ref] for ref in still_pending}
            continue

        ref = ready[0]
        label = pending.pop(ref)
        try:
            results[ref] = ray.get(ref, timeout=1)
        except BaseException as exc:
            for pending_ref in pending:
                ray.cancel(pending_ref, force=True)
            raise RuntimeError(f"Ray ref failed while waiting for {label}") from exc

    return [results[ref] for _, ref in labeled_refs]


def _normalize_labeled_refs(
    refs: Iterable[tuple[str, ray.ObjectRef]] | Iterable[ray.ObjectRef],
) -> list[tuple[str, ray.ObjectRef]]:
    labeled_refs = []
    for index, item in enumerate(refs):
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
            labeled_refs.append(item)
        else:
            labeled_refs.append((f"ref-{index}", item))
    return labeled_refs


@ray.remote(num_cpus=0, max_retries=0, max_calls=1)
def simple_worker(
    *,
    rank: int,
    position: tuple[int, ...],
    chunks_per_dim: tuple[int, ...],
    chunk_size: tuple[int, ...],
    nb_iterations: int,
    nb_nodes: int,
    port: int,
    node_id: str | None = None,
    array_name: str | list[str] = "array",
    dtype: np.dtype = np.int32,  # type: ignore
    _sleep_b4_send=0,
    _sleep_intra_send=0,
    **kwargs,
) -> None:
    """Worker node sending chunks of data"""
    from deisa.ray.bridge import Bridge
    from deisa.ray.comm import init_gloo_comm

    if isinstance(array_name, str):
        array_name = [array_name]

    start_iteration = kwargs.get("start_iteration", 0)

    arrays_md = {
        name: {
            "global_shape": tuple(n * c for n, c in zip(chunks_per_dim, chunk_size)),
            "chunk_shape": chunk_size,
            "chunk_position": position,
        }
        for name in array_name
    }

    comm = init_gloo_comm(
        nb_nodes,
        rank,
        "127.0.0.1",
        port,
    )
    client = Bridge(arrays_metadata=arrays_md, comm=comm, _node_id=node_id)

    array = (rank + 1) * np.ones(chunk_size, dtype=dtype)

    time.sleep(_sleep_b4_send)
    for i in range(start_iteration, nb_iterations):
        time.sleep(_sleep_intra_send)
        for array_described in list(arrays_md.keys()):
            chunk = i * array
            client.send(array_name=array_described, chunk=chunk, timestep=i)

    client.close(timestep=nb_iterations)


@ray.remote(num_cpus=0, max_retries=0, max_calls=1)
def simple_worker_error_test(
    *,
    rank: int,
    position: tuple[int, ...],
    chunks_per_dim: tuple[int, ...],
    chunk_size: tuple[int, ...],
    nb_iterations: int,
    nb_nodes: int,
    port: int,
    node_id: str | None = None,
    array_name: str = "array",
    dtype: np.dtype = np.int32,  # type: ignore
) -> None:
    """Worker node sending chunks of data"""
    from deisa.ray.bridge import Bridge
    from deisa.ray.comm import init_gloo_comm

    arrays_md = {
        array_name: {
            "global_shape": tuple(n * c for n, c in zip(chunks_per_dim, chunk_size)),
            "chunk_shape": chunk_size,
            "chunk_position": position,
        }
    }

    comm = init_gloo_comm(
        nb_nodes,
        rank,
        "127.0.0.1",
        port,
    )
    client = Bridge(arrays_metadata=arrays_md, comm=comm, _node_id=node_id)

    array = (rank + 1) * np.ones(chunk_size, dtype=dtype)

    for i in range(nb_iterations):
        chunk = i * array
        if i == nb_iterations // 2:
            client.send(array_name="error", chunk=chunk, timestep=i)
        else:
            client.send(array_name=array_name, chunk=chunk, timestep=i)
    client.close(timestep=nb_iterations)
