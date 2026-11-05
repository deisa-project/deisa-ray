"""
Gysela-style analytics patterns on deisa-ray.

Ported from the "Gysela-style patterns" section of deisa-dask's ``test_precompute.py``. There, the two callback shapes
Gysela's post-processing uses are checked against the precompute analyzer's extracted reduction hints:

- ``density(f, grid) = da.sum(f, axis=(0, 3, 4)) * grid.dvx * grid.dvy`` -- one reduction over a 5-D distribution
  ``(species, x, y, vx, vy)``, collapsing the species and velocity axes;
- ``measure(cfg, f)`` -- five INDEPENDENT reductions (energy, L2 norm, mass, and two momentum fluxes) of one species'
  distribution, called once per species in a loop.

deisa-ray has no precompute analyzer: the callback runs in the analytics process on the dask array assembled from the
bridges' chunks. So the analogous test runs the SAME two helpers end to end through
:class:`deisa.ray.window_handler.Deisa` and real :class:`deisa.ray.bridge.Bridge` instances, splitting the 5-D
distribution across two bridges on the species axis, and checks the NUMBERS against a numpy reference instead of
checking extracted reduction hints.
"""

import numpy as np
import pytest
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from deisa.ray.bridge import Bridge
from deisa.ray.types import DeisaArray
from tests.comm_utils import NoOpComm
from tests.utils import wait_for_head_node

# The distribution shape and its split: two bridges each own two species, with the whole (x, y, vx, vy) block on both.
SHAPE = (4, 5, 3, 4, 2)  # (species, x, y, vx, vy)
CHUNK = (2, 5, 3, 4, 2)
N_BRIDGES = SHAPE[0] // CHUNK[0]

# Grid weights and the velocity abscissae the measure() helper broadcasts against.
DVX = 0.5
DVY = 0.25
DV_4D = 0.125


def _field() -> np.ndarray:
    """The full distribution, built deterministically so both the bridges and the head agree on it."""
    return np.arange(int(np.prod(SHAPE)), dtype=np.float64).reshape(SHAPE)


@ray.remote(max_retries=0)
def head_script(enable_distributed_scheduling: bool) -> None:
    """Register the Gysela-style callback and check its numbers on the assembled array."""
    import os
    from types import SimpleNamespace

    import dask.array as da

    from deisa.ray.window_handler import Deisa

    os.environ["DEISA_DISTRIBUTED_SCHEDULING"] = "1" if enable_distributed_scheduling else "0"

    d = Deisa()

    reference = _field()
    vx = np.linspace(-1.0, 1.0, SHAPE[3])
    vy = np.linspace(-2.0, 2.0, SHAPE[4])

    # Stand-in for the Gysela grid/config object: volume weights plus the velocity abscissae.
    grid = SimpleNamespace(dvx=DVX, dvy=DVY, dV_4D=DV_4D, vx=vx, vy=vy)

    def density(f, cfg):
        """Gysela density helper: reduce the 5-D distribution, then apply the volume weights."""
        return da.sum(f, axis=(0, 3, 4)) * cfg.dvx * cfg.dvy

    def measure(cfg, f):
        """Gysela measure helper: five independent reductions of one species' 4-D distribution."""
        vx_bc = cfg.vx[(None, None, slice(None), None)]
        vy_bc = cfg.vy[(None, None, None, slice(None))]
        v2 = vx_bc**2 + vy_bc**2
        return {
            "ek": 0.5 * da.sum(f * v2) * cfg.dV_4D,
            "l2": da.sum(f**2) * cfg.dV_4D,
            "ms": da.sum(f) * cfg.dV_4D,
            "mx": da.sum(f * vx_bc) * cfg.dV_4D,
            "my": da.sum(f * vy_bc) * cfg.dV_4D,
        }

    @d.register("fdistribu")
    def gysela_callback(fdistribu: list[DeisaArray]):
        field = fdistribu[0]

        # Pattern 1 -- density(f, grid): the delivered 5-D array reduced to (x, y).
        expected_density = reference.sum(axis=(0, 3, 4)) * DVX * DVY
        np.testing.assert_allclose(density(field, grid).compute(), expected_density, rtol=1e-12, atol=1e-12)

        # Pattern 2 -- measure(client, cfg, f) called once per species.
        vx_bc = vx[None, None, :, None]
        vy_bc = vy[None, None, None, :]
        for isp in range(SHAPE[0]):
            got = {name: value.compute() for name, value in measure(grid, field[isp]).items()}
            f_ref = reference[isp]
            expected = {
                "ek": 0.5 * np.sum(f_ref * (vx_bc**2 + vy_bc**2)) * DV_4D,
                "l2": np.sum(f_ref**2) * DV_4D,
                "ms": np.sum(f_ref) * DV_4D,
                "mx": np.sum(f_ref * vx_bc) * DV_4D,
                "my": np.sum(f_ref * vy_bc) * DV_4D,
            }
            for name, value in expected.items():
                np.testing.assert_allclose(got[name], value, rtol=1e-12, atol=1e-12)

    d.execute_callbacks()


@ray.remote(max_retries=0)
def bridge_script(rank: int) -> str:
    """Send this bridge's species slab of the distribution."""
    field = _field()
    arrays_md = {
        "fdistribu": {
            "global_shape": SHAPE,
            "chunk_shape": CHUNK,
            "chunk_position": (rank, 0, 0, 0, 0),
        }
    }

    bridge = Bridge(
        arrays_metadata=arrays_md,
        comm=NoOpComm(rank, N_BRIDGES),
    )  # type: ignore[arg-type]

    bridge.send(
        array_name="fdistribu",
        chunk=field[rank * CHUNK[0] : (rank + 1) * CHUNK[0]],
        timestep=0,
    )

    return bridge.node_id


@pytest.mark.parametrize(
    "enable_distributed_scheduling",
    [False, True],
)
def test_gysela_style_patterns(enable_distributed_scheduling, ray_multinode_cluster):
    cluster = ray_multinode_cluster["cluster"]
    head_node_id = None
    worker_node_ids = []
    for node in cluster.list_all_nodes():
        if node.is_head():
            head_node_id = node.node_id
        else:
            worker_node_ids.append(node.node_id)

    head_ref = head_script.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=head_node_id,
            soft=False,
        ),
    ).remote(enable_distributed_scheduling)
    wait_for_head_node()

    # Rank 0 must be sent LAST: rank 0 owns the close sentinel, and sending it first would tell analytics the
    # simulation is over before the distribution is complete. test_analytics.py orders its bridges the same way.
    for rank in reversed(range(N_BRIDGES)):
        ray.get(
            bridge_script.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=worker_node_ids[rank],
                    soft=False,
                ),
            ).remote(rank)
        )

    ray.get(head_ref)
