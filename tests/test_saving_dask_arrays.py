# TODO use deisa.core DeisaArray and these tests should be moved to deisa.core/tests/test_saving_dask_arrays.py
import os
import pathlib
from typing import TypedDict

import dask.array as da
import numpy as np
import pytest
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from deisa.ray.types import DeisaArray
from tests.utils import pick_free_port, wait_for_head_node

NB_ITERATIONS = 6
EXPECTED_AT_T5 = 5 * np.array([[1, 2]])


class PersistencePaths(TypedDict):
    hdf5: list[str]
    hdf5_timesteps: list[str]
    hdf5_arrays: str
    zarr: list[str]
    netcdf: str


@ray.remote(max_retries=0)
def head_script(paths: PersistencePaths, enable_distributed_scheduling: bool) -> None:
    """The head node saves data through Dask-backed DEISA arrays."""
    from deisa.ray.window_handler import Deisa

    os.environ["DEISA_DISTRIBUTED_SCHEDULING"] = "1" if enable_distributed_scheduling else "0"

    d = Deisa()

    import xarray as xr

    from deisa.ray.types import to_hdf5

    @d.register("array")
    def save_single_array_outputs(array: list[DeisaArray]):
        for fname in paths["hdf5_timesteps"]:
            array[0].to_hdf5(fname, str(array[0].t))

        if array[0].t != 5:
            return

        for fname in paths["hdf5"]:
            array[0].to_hdf5(fname, "data")

        for fname in paths["zarr"]:
            array[0].to_zarr(fname, component="data")

        xarray_da = xr.DataArray(array[0], dims=["x", "y"], name="data").compute()
        xarray_da.to_netcdf(paths["netcdf"])

    @d.register("a", "b")
    def save_multiple_array_hdf5(a: list[DeisaArray], b: list[DeisaArray]):
        if a[0].t == 5:
            to_hdf5(paths["hdf5_arrays"], {"a": a[0], "b": b[0]})

    d.execute_callbacks()


@ray.remote(num_cpus=0, max_retries=0)
def bridge_script(*, rank: int, port: int, array_names: list[str]) -> None:
    from deisa.ray.bridge import Bridge
    from tests.comm_utils import init_gloo_comm

    arrays_md = {
        name: {
            "global_shape": (1, 2),
            "chunk_shape": (1, 1),
            "chunk_position": (0, rank),
        }
        for name in array_names
    }

    comm = init_gloo_comm(
        2,
        rank,
        "127.0.0.1",
        port,
    )
    bridge = Bridge(arrays_metadata=arrays_md, comm=comm, _node_id=f"node_{rank}")

    array = (rank + 1) * np.ones((1, 1), dtype=np.int32)
    for i in range(NB_ITERATIONS):
        for array_name in array_names:
            bridge.send(array_name=array_name, chunk=i * array, timestep=i)

    bridge.close(timestep=NB_ITERATIONS)


def _node_ids(ray_multinode_cluster) -> tuple[str, list[str]]:
    cluster = ray_multinode_cluster["cluster"]
    head_node_id = None
    worker_node_ids = []
    for node in cluster.list_all_nodes():
        if node.is_head():
            head_node_id = node.node_id
        else:
            worker_node_ids.append(node.node_id)

    assert head_node_id is not None
    assert len(worker_node_ids) == 2
    return head_node_id, worker_node_ids


def _run_save_workflow(
    *,
    ray_multinode_cluster,
    paths: PersistencePaths,
    enable_distributed_scheduling: bool,
) -> None:
    head_node_id, worker_node_ids = _node_ids(ray_multinode_cluster)
    head_ref = head_script.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=head_node_id,
            soft=False,
        ),
    ).remote(paths, enable_distributed_scheduling)
    wait_for_head_node()
    port = pick_free_port()

    array_names = ["array", "a", "b"]

    worker_refs = [
        bridge_script.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=worker_node_ids[rank],
                soft=False,
            ),
        ).remote(rank=rank, port=port, array_names=array_names)
        for rank in range(2)
    ]

    ray.get([head_ref] + worker_refs)


def _output_path(tmp_path: pathlib.Path, fname: str) -> pathlib.Path:
    """Return an isolated path while preserving relative and tilde-shaped cases."""
    if fname.startswith("~/"):
        return tmp_path / "home" / fname.removeprefix("~/")
    return tmp_path / fname


def _persistence_paths(tmp_path: pathlib.Path) -> PersistencePaths:
    hdf5_paths = [
        _output_path(tmp_path, "interesting-event.h5"),
        _output_path(tmp_path, "~/interesting-event.h5"),
    ]
    hdf5_timesteps_paths = [_output_path(tmp_path, "timesteps.h5")]
    hdf5_arrays_path = _output_path(tmp_path, "several-arrays.h5")
    zarr_paths = [
        _output_path(tmp_path, "interesting-event.zarr"),
        _output_path(tmp_path, "~/interesting-event.zarr"),
    ]
    netcdf_path = _output_path(tmp_path, "interesting-event.nc")

    for output_path in [*hdf5_paths, *hdf5_timesteps_paths, hdf5_arrays_path, *zarr_paths, netcdf_path]:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    return {
        "hdf5": [str(path) for path in hdf5_paths],
        "hdf5_timesteps": [str(path) for path in hdf5_timesteps_paths],
        "hdf5_arrays": str(hdf5_arrays_path),
        "zarr": [str(path) for path in zarr_paths],
        "netcdf": str(netcdf_path),
    }


def _assert_hdf5_dataset(path: pathlib.Path, dataset: str, expected: np.ndarray) -> None:
    import h5py

    with h5py.File(path) as h5file:
        data = da.from_array(h5file[dataset], chunks=2)
        assert (data.compute() == expected).all()


@pytest.mark.parametrize("enable_distributed_scheduling", [False, True])
def test_dask_array_persistence_formats(enable_distributed_scheduling, ray_multinode_cluster, tmp_path) -> None:  # noqa: F811
    import h5py
    import xarray as xr

    paths = _persistence_paths(tmp_path)

    _run_save_workflow(
        ray_multinode_cluster=ray_multinode_cluster,
        paths=paths,
        enable_distributed_scheduling=enable_distributed_scheduling,
    )

    for hdf5_path in paths["hdf5"]:
        _assert_hdf5_dataset(pathlib.Path(hdf5_path), "data", EXPECTED_AT_T5)

    with h5py.File(paths["hdf5_timesteps"][0]) as h5file:
        for i in range(NB_ITERATIONS):
            data = da.from_array(h5file[str(i)], chunks=2)
            arr = i * np.array([[1, 2]])
            assert (data.compute() == arr).all()

    _assert_hdf5_dataset(pathlib.Path(paths["hdf5_arrays"]), "a", EXPECTED_AT_T5)
    _assert_hdf5_dataset(pathlib.Path(paths["hdf5_arrays"]), "b", EXPECTED_AT_T5)

    for zarr_path in paths["zarr"]:
        data = da.from_zarr(zarr_path, component="data")
        assert (data.compute() == EXPECTED_AT_T5).all()

    with xr.open_dataarray(paths["netcdf"]) as data:
        assert (data.compute() == EXPECTED_AT_T5).all()
        assert data.dims == ("x", "y")


# Regression note:
# test_feedback_loop previously leaked a Dask scheduler change into this file.
# The leak happened in the pytest driver process: Deisa.set() called
# _ensure_connected(), which globally set Dask's scheduler to ray_dask_get. The
# DEISA_DISTRIBUTED_SCHEDULING is set in this file inside
# Ray remote head_script tasks, so they configure those Ray worker processes, not
# the pytest driver process that later opens HDF5/Zarr output and calls
# data.sum().compute() or data.compute().
#
# With the leaked ray_dask_get scheduler, these driver-side HDF5/Zarr Dask graphs
# were submitted to Ray workers instead of running with Dask's normal local
# scheduler. Dask-on-Ray is valid for many graphs, but these file-backed graphs
# appear to break on the Ray serialization/execution path in this environment
# while working with the normal Dask scheduler. That interaction is why
# window_handler.py now scopes the Ray scheduler with a context manager around
# execute_callbacks() only, rather than setting it globally during Deisa
# connection or feedback publication.
