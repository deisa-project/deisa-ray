import ray
import ray.util.dask.scheduler

@ray.remote(num_cpus=0, enable_task_events=False)
def patched_dask_task_wrapper(func, repack, key, ray_pretask_cbs, ray_posttask_cbs, *args, first_call=True):
    """
    Patched version of the original dask_task_wrapper function.

    This function wraps Dask tasks to handle Ray ObjectRefs properly. It
    receives ObjectRefs first, then calls itself a second time with
    num_cpus=1 to unwrap the ObjectRefs and execute the actual computation.

    Parameters
    ----------
    func : Callable
        The Dask task function to execute.
    repack : Callable
        Function to repack arguments and dependencies.
    key : Any
        The task key in the Dask graph.
    ray_pretask_cbs : list[Callable] or None
        List of pre-task callbacks to execute before the task.
    ray_posttask_cbs : list[Callable] or None
        List of post-task callbacks to execute after the task.
    *args
        Arguments to pass to the task function. On first call, these are
        ObjectRefs. On second call, these are the unwrapped values.
    first_call : bool, optional
        If True, this is the first call and ObjectRefs need to be unwrapped.
        If False, execute the actual task. Default is True.

    Returns
    -------
    ray.ObjectRef
        On the first call, returns an ObjectRef to the second call. On the
        second call, returns the result of executing the task function.

    Notes
    -----
    This is a two-phase execution: first call schedules the second call with
    CPU resources, second call unwraps ObjectRefs and executes the task.
    This allows proper resource allocation for Dask tasks in Ray.
    """

    if first_call:
        assert all([isinstance(a, ray.ObjectRef) for a in args])
        # Use one CPU for the actual computation
        return patched_dask_task_wrapper.options(num_cpus=1).remote(
            func, repack, key, ray_pretask_cbs, ray_posttask_cbs, *args, first_call=False
        )

    if ray_pretask_cbs is not None:
        pre_states = [cb(key, args) if cb is not None else None for cb in ray_pretask_cbs]
    repacked_args, repacked_deps = repack(args)
    # Recursively execute Dask-inlined tasks.
    actual_args = [ray.util.dask.scheduler._execute_task(a, repacked_deps) for a in repacked_args]
    # Execute the actual underlying Dask task.
    result = func(*actual_args)

    if ray_posttask_cbs is not None:
        for cb, pre_state in zip(ray_posttask_cbs, pre_states):
            if cb is not None:
                cb(key, result, pre_state)

    return result


@ray.remote(num_cpus=0, enable_task_events=False)
def remote_ray_dask_get(dsk, keys):
    """
    Execute a Dask task graph using Ray with a patched task wrapper.

    This function monkey-patches Ray's Dask scheduler to use the patched
    task wrapper, then executes the task graph and returns the results.

    Parameters
    ----------
    dsk : dict
        The Dask task graph dictionary.
    keys : list
        List of keys to compute from the task graph.

    Returns
    -------
    list[ray.ObjectRef]
        List of Ray object references to the computed results.

    Notes
    -----
    This function patches `ray.util.dask.scheduler.dask_task_wrapper` with
    `patched_dask_task_wrapper` to enable proper resource allocation for
    Dask tasks. The `ray_persist=True` option ensures results are kept in
    Ray's object store.
    """
    import ray.util.dask

    # Monkey-patch Dask-on-Ray
    ray.util.dask.scheduler.dask_task_wrapper = patched_dask_task_wrapper

    return ray.util.dask.ray_dask_get(dsk, keys, ray_persist=True)
