import os

import numpy as np
import pytest
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from deisa.ray.bridge import Bridge
from deisa.ray.comm import NoOpComm
from deisa.ray.types import DeisaArray
from tests.utils import ray_multinode_cluster, wait_for_head_node  # noqa: F401


@ray.remote(max_retries=0)
def head_script(enable_distributed_scheduling: bool) -> None:
    """Run all analytics checks from a single Deisa callback."""
    import dask.array as da

    from deisa.ray.window_handler import Deisa

    os.environ["DEISA_DISTRIBUTED_SCHEDULING"] = "1" if enable_distributed_scheduling else "0"

    d = Deisa()

    @d.register("array")
    def analytics_callback(array: list[DeisaArray]):
        d_arr = array[0]

        computed = d_arr.compute()
        print(f"analytics array = {computed}", flush=True)
        np.testing.assert_array_equal(computed, np.array([[1, 2]], dtype=np.int64))

        assert isinstance(d_arr.mean().compute(), float)
        assert d_arr.mean().compute() == 1.5
        assert d_arr.sum().compute() == 3
        assert d_arr.min().compute() == 1
        assert d_arr.max().compute() == 2
        assert d_arr.std().compute() == 0.5

        multiplied_by_int = (d_arr * 2).compute()
        np.testing.assert_array_equal(
            multiplied_by_int,
            np.array([[2, 4]], dtype=np.int64),
        )

        multiplied_between_arrays = (d_arr * d_arr).compute()
        np.testing.assert_array_equal(
            multiplied_between_arrays,
            np.array([[1, 4]], dtype=np.int64),
        )

        sliced = d_arr[:, :1].compute()
        np.testing.assert_array_equal(sliced, np.array([[1]], dtype=np.int64))

        transposed = d_arr.T.compute()
        np.testing.assert_array_equal(
            transposed,
            np.array([[1], [2]], dtype=np.int64),
        )

        matrix_product = (d_arr.T @ d_arr).compute()
        np.testing.assert_array_equal(
            matrix_product,
            np.array([[1, 2], [2, 4]], dtype=np.int64),
        )

        clipped = d_arr.clip(1, 1).compute()
        np.testing.assert_array_equal(clipped, np.array([[1, 1]], dtype=np.int64))

        masked = da.where(d_arr > 1, d_arr, 0).compute()
        np.testing.assert_array_equal(masked, np.array([[0, 2]], dtype=np.int64))

        square_root = da.sqrt(d_arr.astype(float)).compute()
        np.testing.assert_allclose(square_root, np.array([[1.0, np.sqrt(2.0)]]))

        stacked = da.concatenate([d_arr, d_arr * 2, d_arr * 3], axis=0).rechunk(
            (3, 2)
        )
        np.testing.assert_array_equal(
            stacked.compute(),
            np.array([[1, 2], [2, 4], [3, 6]], dtype=np.int64),
        )

        centered = stacked.astype(float) - stacked.mean(axis=0)
        _, singular_values, principal_components = da.linalg.svd(centered)
        np.testing.assert_allclose(
            singular_values.compute(),
            np.array([np.sqrt(10.0), 0.0]),
            atol=1e-12,
        )
        first_component = np.abs(principal_components.compute()[0])
        np.testing.assert_allclose(
            first_component,
            np.array([1 / np.sqrt(5), 2 / np.sqrt(5)]),
        )

        fft = da.fft.fft(d_arr.astype(float).rechunk((1, 2)), axis=1).compute()
        np.testing.assert_allclose(fft, np.array([[3.0 + 0.0j, -1.0 + 0.0j]]))

    d.execute_callbacks()


@ray.remote(max_retries=0)
def bridge_script(rank: int) -> str:
    arrays_md = {
        "array": {
            "global_shape": (1, 2),
            "chunk_shape": (1, 1),
            "chunk_position": (0, rank),
        }
    }

    bridge = Bridge(
        arrays_metadata=arrays_md,
        comm=NoOpComm(rank, 2),
    )  # type:ignore

    bridge.send(
        array_name="array",
        chunk=np.array([[rank + 1]], dtype=np.int64),
        timestep=0,
    )

    return bridge.node_id


@pytest.mark.parametrize(
    "enable_distributed_scheduling",
    [False, True],
)
def test_deisa_analytics(enable_distributed_scheduling, ray_multinode_cluster):
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

    ray.get(
        bridge_script.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=worker_node_ids[1],
                soft=False,
            ),
        ).remote(1)
    )
    ray.get(
        bridge_script.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=worker_node_ids[0],
                soft=False,
            ),
        ).remote(0)
    )

    ray.get(head_ref)
