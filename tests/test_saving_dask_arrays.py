# TODO use deisa.core DeisaArray and these tests should be moved to deisa.core/tests/test_saving_dask_arrays.py
import os
import pathlib

import dask.array as da
import numpy as np
import pytest
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from deisa.ray.types import DeisaArray
from tests.utils import pick_free_port, ray_multinode_cluster, wait_for_head_node  # noqa: F401

NB_ITERATIONS = 6
EXPECTED_AT_T5 = 5 * np.array([[1, 2]])


@ray.remote(max_retries=0)
def head_script(fname: str, enable_distributed_scheduling: bool, save_mode: str) -> None:
    """The head node saves data through Dask-backed DEISA arrays."""
    from deisa.ray.window_handler import Deisa

    os.environ["DEISA_DISTRIBUTED_SCHEDULING"] = "1" if enable_distributed_scheduling else "0"

    d = Deisa()

    if save_mode == "hdf5":

        @d.register("array")
        def simulation_callback(array: list[DeisaArray]):
            if array[0].t == 5:
                array[0].to_hdf5(fname, "data")

    elif save_mode == "hdf5_timesteps":

        @d.register("array")
        def simulation_callback(array: list[DeisaArray]):
            array[0].to_hdf5(fname, str(array[0].t))

    elif save_mode == "hdf5_arrays":
        from deisa.ray.types import to_hdf5

        @d.register("a", "b")
        def simulation_callback(a: list[DeisaArray], b: list[DeisaArray]):
            if a[0].t == 5:
                to_hdf5(fname, {"a": a[0], "b": b[0]})

    elif save_mode == "zarr":

        @d.register("array")
        def simulation_callback(array: list[DeisaArray]):
            if array[0].t == 5:
                array[0].to_zarr(fname, component="data")

    elif save_mode == "netcdf":
        import xarray as xr

        @d.register("array")
        def simulation_callback(array: list[DeisaArray]):
            if array[0].t == 5:
                xarray_da = xr.DataArray(array[0], dims=["x", "y"], name="data").compute()
                xarray_da.to_netcdf(fname)

    else:
        raise ValueError(f"Unknown save mode: {save_mode}")

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
    fname: str,
    enable_distributed_scheduling: bool,
    save_mode: str,
) -> None:
    head_node_id, worker_node_ids = _node_ids(ray_multinode_cluster)
    head_ref = head_script.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=head_node_id,
            soft=False,
        ),
    ).remote(fname, enable_distributed_scheduling, save_mode)
    wait_for_head_node()
    port = pick_free_port()

    array_names = ["a", "b"] if save_mode == "hdf5_arrays" else ["array"]

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


@pytest.mark.parametrize(
    "fname, enable_distributed_scheduling",
    [
        ("interesting-event.h5", False),
        ("interesting-event.h5", True),
        ("~/interesting-event.h5", True),
        ("~/interesting-event.h5", False),
    ],
)
def test_dask_save_hdf5(
    fname, enable_distributed_scheduling, ray_multinode_cluster, tmp_path
) -> None:  # noqa: F811
    import h5py

    full_name = _output_path(tmp_path, fname)
    full_name.parent.mkdir(parents=True, exist_ok=True)

    _run_save_workflow(
        ray_multinode_cluster=ray_multinode_cluster,
        fname=str(full_name),
        enable_distributed_scheduling=enable_distributed_scheduling,
        save_mode="hdf5",
    )

    with h5py.File(full_name) as h5file:
        data = da.from_array(h5file["data"], chunks=2)
        assert (data.compute() == EXPECTED_AT_T5).all()


@pytest.mark.parametrize(
    "fname, enable_distributed_scheduling",
    [
        ("interesting-event.h5", False),
        ("interesting-event.h5", True),
    ],
)
def test_dask_save_several_timesteps_hdf5(
    fname, enable_distributed_scheduling, ray_multinode_cluster, tmp_path
) -> None:  # noqa: F811
    import h5py

    full_name = _output_path(tmp_path, fname)
    full_name.parent.mkdir(parents=True, exist_ok=True)

    _run_save_workflow(
        ray_multinode_cluster=ray_multinode_cluster,
        fname=str(full_name),
        enable_distributed_scheduling=enable_distributed_scheduling,
        save_mode="hdf5_timesteps",
    )

    with h5py.File(full_name) as h5file:
        for i in range(NB_ITERATIONS):
            data = da.from_array(h5file[str(i)], chunks=2)

            arr = i * np.array([[1, 2]])
            assert (data.compute() == arr).all()


@pytest.mark.parametrize(
    "fname, enable_distributed_scheduling",
    [
        ("interesting-event.h5", False),
        ("interesting-event.h5", True),
    ],
)
def test_dask_save_several_arrays_hdf5(
    fname, enable_distributed_scheduling, ray_multinode_cluster, tmp_path
) -> None:  # noqa: F811
    import h5py

    full_name = _output_path(tmp_path, fname)
    full_name.parent.mkdir(parents=True, exist_ok=True)

    _run_save_workflow(
        ray_multinode_cluster=ray_multinode_cluster,
        fname=str(full_name),
        enable_distributed_scheduling=enable_distributed_scheduling,
        save_mode="hdf5_arrays",
    )

    with h5py.File(full_name) as h5file:
        data_a = da.from_array(h5file["a"], chunks=2)
        assert (data_a.compute() == EXPECTED_AT_T5).all()

        data_b = da.from_array(h5file["b"], chunks=2)
        assert (data_b.compute() == EXPECTED_AT_T5).all()


@pytest.mark.parametrize(
    "fname, enable_distributed_scheduling",
    [
        ("interesting-event.zarr", False),
        ("~/interesting-event.zarr", False),
        ("interesting-event.zarr", True),
        ("~/interesting-event.zarr", True),
    ],
)
def test_dask_save_zarr(
    fname, enable_distributed_scheduling, ray_multinode_cluster, tmp_path
) -> None:  # noqa: F811
    full_path = _output_path(tmp_path, fname)
    full_path.parent.mkdir(parents=True, exist_ok=True)

    _run_save_workflow(
        ray_multinode_cluster=ray_multinode_cluster,
        fname=str(full_path),
        enable_distributed_scheduling=enable_distributed_scheduling,
        save_mode="zarr",
    )

    data = da.from_zarr(full_path, component="data")
    assert (data.compute() == EXPECTED_AT_T5).all()


@pytest.mark.parametrize(
    "fname, enable_distributed_scheduling",
    [
        ("interesting-event.nc", False),
        ("interesting-event.nc", True),
    ],
)
def test_dask_save_netcdf_xarray(
    fname, enable_distributed_scheduling, ray_multinode_cluster, tmp_path
) -> None:  # noqa: F811
    import xarray as xr

    full_name = _output_path(tmp_path, fname)
    full_name.parent.mkdir(parents=True, exist_ok=True)

    _run_save_workflow(
        ray_multinode_cluster=ray_multinode_cluster,
        fname=str(full_name),
        enable_distributed_scheduling=enable_distributed_scheduling,
        save_mode="netcdf",
    )

    with xr.open_dataarray(full_name) as data:
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
