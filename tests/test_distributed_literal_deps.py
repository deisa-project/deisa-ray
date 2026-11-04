"""End-to-end regression: chunk data mixed with COMPUTED CONSTANTS under distributed scheduling.

A realistic callback builds small arrays (``da.fft.fftfreq``, ``da.ones_like``, ``k2[0, 0] = 1.0``, ...) and combines
them with the distributed field. In the graph these constants are dependency-free literals (``DataNode`` / ``Alias``),
not chunks. ``ray_dask_get(..., ray_persist=True)`` returns a ref-of-ref for tasks but a PLAIN VALUE for such literals,
which used to break two things in the distributed scheduler:

* ``SchedulingActor.get_value`` awaited the plain value -> ``TypeError: object numpy.ndarray can't be used in 'await'``;
* a literal needed by a task on another actor was routed through ``get_value``, so dask tasks such as ``setitem``
  received a ``ray.ObjectRef`` where a number was expected.

Both are intermittent (they depend on where the partitioner happened to place the literal), which is why this pattern
could run for a couple of timesteps and then stall. This test drives the real path -- two bridges, a distributed
callback mixing a chunked field with an initialised-from-timestep-0 grid and FFTs, several timesteps -- and asserts
that every timestep's callback actually completed.
"""

from __future__ import annotations

import numpy as np
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from tests.utils import pick_free_port, wait_for_head_node

DIAG_STEPS = [0, 2, 4, 6]


@ray.remote(max_retries=0)
def head_script() -> int:
    """Run the app-like callback for every diag timestep and return how many completed."""
    import os

    import dask.array as da

    from deisa.ray.window_handler import Deisa

    os.environ["DEISA_DISTRIBUTED_SCHEDULING"] = "1"
    d = Deisa()

    grid = {}

    def _fail_fast(exc: BaseException) -> None:
        # The default handler logs and continues, so a broken callback would not fail the test.
        raise exc

    @d.register(
        "fdistribu",
        "absolute_time",
        "deltat",
        "MeshX",
        "MeshY",
        "MeshVx",
        "MeshVy",
        exception_handler=_fail_fast,
    )
    def compute_diagnostics(fdistribu, absolute_time, deltat, MeshX, MeshY, MeshVx, MeshVy):
        if "x" not in grid:
            # Initialise once from timestep 0, then keep using the stored grid (like the real diagnostics).
            x, y = np.asarray(MeshX[0]), np.asarray(MeshY[0])
            vx, vy = np.asarray(MeshVx[0]), np.asarray(MeshVy[0])
            grid.update(x=x, y=y, vx=vx, vy=vy, dv=(x[1] - x[0]) * (y[1] - y[0]) * (vx[1] - vx[0]) * (vy[1] - vy[0]))

        f = fdistribu[0]

        # Literal-producing operations: fftfreq / ones_like / scalar setitem all create dependency-free nodes.
        n = da.sum(f, axis=(0, 3, 4))
        rho_hat = da.fft.fft2(n - 1.0)
        kx = da.fft.fftfreq(n.shape[0])
        ky = da.fft.fftfreq(n.shape[1])
        KX, KY = da.meshgrid(kx, ky, indexing="ij")
        k2 = KX**2 + KY**2
        k2[0, 0] = 1.0
        mask = da.ones_like(k2)
        mask[0, 0] = 0.0
        phi = da.real(da.fft.ifft2(rho_hat / k2 * mask))

        vx_bc = grid["vx"][(None, None, slice(None), None)]
        vy_bc = grid["vy"][(None, None, None, slice(None))]
        ekin = 0.5 * da.sum(f * (vx_bc**2 + vy_bc**2)) * grid["dv"]
        mass = da.sum(f) * grid["dv"]
        epot = 0.5 * da.sum(phi**2)

        ekin_v, mass_v, epot_v = da.compute(ekin, mass, epot)
        assert np.isfinite(ekin_v) and np.isfinite(mass_v) and np.isfinite(epot_v)
        grid["completed"] = grid.get("completed", 0) + 1

    d.execute_callbacks()
    return grid.get("completed", 0)


@ray.remote(num_cpus=0, max_retries=0)
def bridge_script(*, rank: int, port: int) -> None:
    from deisa.ray.bridge import Bridge
    from tests.comm_utils import init_gloo_comm

    comm = init_gloo_comm(2, rank, "127.0.0.1", port)

    arrays_md = {
        "fdistribu": {
            "global_shape": (2, 8, 8, 4, 4),
            "chunk_shape": (1, 8, 8, 4, 4),
            "chunk_position": (rank, 0, 0, 0, 0),
        }
    }
    if rank == 0:
        arrays_md.update(
            {
                "absolute_time": {"global_shape": (1,), "chunk_shape": (1,), "chunk_position": (0,)},
                "deltat": {"global_shape": (1,), "chunk_shape": (1,), "chunk_position": (0,)},
                "MeshX": {"global_shape": (8,), "chunk_shape": (8,), "chunk_position": (0,)},
                "MeshY": {"global_shape": (8,), "chunk_shape": (8,), "chunk_position": (0,)},
                "MeshVx": {"global_shape": (4,), "chunk_shape": (4,), "chunk_position": (0,)},
                "MeshVy": {"global_shape": (4,), "chunk_shape": (4,), "chunk_position": (0,)},
            }
        )

    bridge = Bridge(arrays_metadata=arrays_md, comm=comm, _node_id=f"node_{rank}")

    for ts in DIAG_STEPS:
        slab = np.full((1, 8, 8, 4, 4), float(ts + rank), dtype=np.float64)
        bridge.send("fdistribu", slab, ts)
        if rank == 0:
            bridge.send("absolute_time", np.array([0.25 * ts]), ts)
            bridge.send("deltat", np.array([0.25]), ts)
            bridge.send("MeshX", np.linspace(0, 1, 8), ts)
            bridge.send("MeshY", np.linspace(0, 1, 8), ts)
            bridge.send("MeshVx", np.linspace(-1, 1, 4), ts)
            bridge.send("MeshVy", np.linspace(-1, 1, 4), ts)

    bridge.close(timestep=DIAG_STEPS[-1] + 1)


def test_literal_dependencies_survive_distributed_scheduling(ray_multinode_cluster) -> None:
    cluster = ray_multinode_cluster["cluster"]
    head_node_id = None
    worker_node_ids = []
    for node in cluster.list_all_nodes():
        if node.is_head():
            head_node_id = node.node_id
        else:
            worker_node_ids.append(node.node_id)

    head_ref = head_script.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=head_node_id, soft=False)
    ).remote()
    wait_for_head_node()
    port = pick_free_port()

    worker_refs = [
        bridge_script.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=worker_node_ids[rank], soft=False)
        ).remote(rank=rank, port=port)
        for rank in range(2)
    ]

    results = ray.get([head_ref] + worker_refs)
    assert results[0] == len(DIAG_STEPS), f"only {results[0]} of {len(DIAG_STEPS)} callbacks completed"
