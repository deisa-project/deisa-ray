import ray
import ray.util.dask.scheduler
from dask._task_spec import Task, DataNode
from collections.abc import Mapping


@ray.remote(num_cpus=0, enable_task_events=False)
def patched_dask_task_wrapper(task, repack, key, ray_pretask_cbs, ray_posttask_cbs, *arg_object_refs, first_call=True):
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
    ray.ObjectRef or Any
        On the first call, returns a Ray ObjectRef pointing to the second
        invocation (which runs with CPU resources). On the second call,
        returns the concrete task result.

    Notes
    -----
    This is a two-phase execution: first call schedules the second call with
    CPU resources, second call unwraps ObjectRefs and executes the task.
    This allows proper resource allocation for Dask tasks in Ray.
    """

    print(f"first_call = {first_call}, ARGS = {arg_object_refs}", flush=True)
    if first_call:
        assert all([isinstance(a, ray.ObjectRef) for a in arg_object_refs])
        # Use one CPU for the actual computation
        return patched_dask_task_wrapper.options(num_cpus=1).remote(
            task, repack, key, ray_pretask_cbs, ray_posttask_cbs, *
            arg_object_refs, first_call=False
        )

    if ray_pretask_cbs is not None:
        pre_states = [
            cb(key, arg_object_refs) if cb is not None else None
            for cb in ray_pretask_cbs
        ]
    (repacked_deps,) = repack(arg_object_refs)
    # De-reference the potentially nested arguments recursively.

    def _dereference_args(x):
        if isinstance(x, Task):
            x.args = _dereference_args(x.args)
            return x
        elif isinstance(x, Mapping):
            return {k: _dereference_args(v) for k, v in x.items()}
        elif isinstance(x, tuple):
            return tuple(_dereference_args(x) for x in x)
        elif isinstance(x, ray.ObjectRef):
            return ray.get(x)
        elif isinstance(x, DataNode):
            if isinstance(x.value, ray.ObjectRef):
                value = ray.get(x.value)
                return DataNode(key=x.key, value=value)
            return x
        else:
            return x

    task = _dereference_args(task)
    result = task(repacked_deps)

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
    tuple[ray.ObjectRef]
        Tuple of *double* Ray ObjectRefs produced by ``ray_dask_get`` with
        ``ray_persist=True``.

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

    # note: ray_dask_get(..., persist = True) return a tuple of ray refs,
    # if set to false, patched_dask_task_wrapper fails for some reason.
    return ray.util.dask.ray_dask_get(dsk, keys, ray_persist=True)
