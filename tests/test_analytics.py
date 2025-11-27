import pytest
import ray
import dask.array as da
from ray.cluster_utils import Cluster
import dask
from ray.util.dask.scheduler import ray_dask_get
import numpy as np


@pytest.fixture
def ray_multinode_cluster():
    cluster_node_ids = {
        "head": "f64704987dec54e6c20445dc6a063ad34de1cd777d5c7e0779d1100a",
        "node1": "f64704987dec54e6c20445dc6a063ad34de1cd777d5c7e0779d1100b",
        "node2": "f64704987dec54e6c20445dc6a063ad34de1cd777d5c7e0779d1100c",
    }

    cluster = Cluster(
        initialize_head=True,
        connect=False,
        head_node_args={
            "num_cpus": 1,
            "env_vars": {"RAY_OVERRIDE_NODE_ID_FOR_TESTING": cluster_node_ids["head"]},
        },
    )

    cluster.add_node(num_cpus=1, env_vars={"RAY_OVERRIDE_NODE_ID_FOR_TESTING": cluster_node_ids["node1"]})
    cluster.add_node(num_cpus=1, env_vars={"RAY_OVERRIDE_NODE_ID_FOR_TESTING": cluster_node_ids["node2"]})

    # Connect driver to this cluster (IMPORTANT)
    ray.init(
        address=cluster.address,
        include_dashboard=False,
        log_to_driver=True,
        ignore_reinit_error=True,
    )

    dask.config.set(scheduler=ray_dask_get)

    yield {
        "cluster": cluster,
        "ids": cluster_node_ids,
        "address": cluster.address,
    }

    ray.shutdown()
    cluster.shutdown()


def test_sum(ray_multinode_cluster):
    d_arr = da.from_array(np.random.randint(0, 1000, size=(64, 64)), chunks=(32, 64))
    m = d_arr.mean().compute()
    assert isinstance(m, float)


def test_print(ray_multinode_cluster):
    d_arr = da.from_array(np.random.randint(0, 1000, size=(64, 64)), chunks=(32, 64))
    m = d_arr.compute()
    assert m is not None


def test_multiplication_by_int(ray_multinode_cluster):
    d_arr = da.from_array(np.random.randint(0, 1000, size=(64, 64)), chunks=(32, 64))
    m = (d_arr * 2).compute()
    assert m is not None


def test_multiplication_between_arrays(ray_multinode_cluster):
    d_arr1 = da.from_array(np.random.randint(0, 1000, size=(64, 64)), chunks=(32, 64))
    d_arr2 = da.from_array(np.random.randint(0, 1000, size=(64, 64)), chunks=(32, 64))
    m = (d_arr1 * d_arr2).compute()
    assert m is not None


def slice(ray_multinode_cluster):
    d_arr = da.from_array(np.random.randint(0, 1000, size=(64, 64)), chunks=(32, 64))
    m = (d_arr[:10, 12:34]).compute()
    assert m is not None


# def test_pca(ray_multinode_cluster):
#     d_arr = da.from_array(np.random.randint(0, 1000, size=(64, 64)), chunks = (32,64) )
#     m = (d_arr*2).compute()
#     assert m is not None
