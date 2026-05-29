from .utils import ray_cluster, ray_workflow  # noqa: F401


def pytest_sessionfinish(session, exitstatus):
    import ray

    if ray.is_initialized():
        ray.shutdown()
