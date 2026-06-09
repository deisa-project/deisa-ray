import dask.array as da
import numpy as np
from tests.utils import ray_multinode_cluster  # noqa: F401


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


def test_pca(ray_multinode_cluster):
    d_arr = da.from_array(np.random.randint(0, 1000, size=(64, 64)), chunks=(32, 64))
    m = (d_arr * 2).compute()
    assert m is not None
