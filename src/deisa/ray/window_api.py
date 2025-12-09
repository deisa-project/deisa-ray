import gc
from typing import Any, Callable, Hashable

import dask
import dask.array as da
import ray

from deisa.ray._scheduler import deisa_ray_get
from deisa.ray.head_node import HeadNodeActor
from deisa.ray.utils import get_head_actor_options
from deisa.ray.types import WindowArrayDefinition, _CallbackConfig
import logging


@ray.remote(num_cpus=0, max_retries=0)
def _call_prepare_iteration(prepare_iteration: Callable, array: da.Array, timestep: int):
    """
    Call the prepare_iteration function with the given array and timestep.

    This is a Ray remote function that executes the prepare_iteration
    callback with a Dask array. It configures Dask to use the Deisa-Ray
    scheduler before calling the function.

    Parameters
    ----------
    prepare_iteration : Callable
        The function to call. Should accept a Dask array and a timestep
        keyword argument.
    array : da.Array
        The Dask array to pass to the prepare_iteration function.
    timestep : int
        The current timestep to pass to the prepare_iteration function.

    Returns
    -------
    Any
        The result of calling `prepare_iteration(array, timestep=timestep)`.

    Notes
    -----
    This function is executed as a Ray remote task with no CPU requirements
    and no retries. It configures Dask to use the Deisa-Ray scheduler before
    executing the prepare_iteration callback.
    """
    dask.config.set(scheduler=deisa_ray_get, shuffle="tasks")
    return prepare_iteration(array, timestep=timestep)


class Deisa:

    def __init__(self,
                 ):
        """
        Initialize Ray and configure Dask to use the Deisa-Ray scheduler.

        This function initializes Ray if it hasn't been initialized yet, and
        configures Dask to use the Deisa-Ray custom scheduler with task-based
        shuffling.

        Notes
        -----
        Ray is initialized with automatic address detection, logging to driver
        disabled, and error-level logging. Dask is configured to use the
        `deisa_ray_get` scheduler with task-based shuffling.
        """
        if not ray.is_initialized():
            ray.init(address="auto", log_to_driver=False, logging_level=logging.ERROR)

        dask.config.set(scheduler=deisa_ray_get, shuffle="tasks")
        # create HeadNodeActor that coordinates everything.
        self.head: Any = HeadNodeActor.options(**get_head_actor_options()).remote()
        # store all node actors
        self.node_actors = {}
        # store registered callbacks from user analytics
        self.registered_callbacks: list[_CallbackConfig] = []

    # TODO add persist 
    def set(self,
            *args,
            key: Hashable,
            value: Any,
            chunked: bool = False,
            **kwargs
            )->None:
        # TODO test
        if not self.node_actors:
            # retrieve node actors at least once
            self.node_actors = ray.get(self.head.list_scheduling_actors.remote())

        if not chunked:
            for _, handle in self.node_actors.items():
                # set the value inside each node actor
                # TODO: does it need to be blocking? 
                handle.set.remote(key,value,chunked)
        else:
            # TODO: implement chunked version
            raise NotImplementedError()

    def unregister_callback(
        self,
        simulation_callback: Callable,
    )->None:
        raise NotImplementedError("method not yet implemented.")

    def register_callback(
        self,
        simulation_callback: Callable,
        arrays_description: list[WindowArrayDefinition],
        *,
        max_iterations=1000_000_000,
        prepare_iteration: Callable | None = None,
        preparation_advance: int = 3,
    ) -> None:
        cfg = _CallbackConfig(
            simulation_callback=simulation_callback,
            arrays_description=arrays_description,
            max_iterations=max_iterations,
            prepare_iteration=prepare_iteration,
            preparation_advance=preparation_advance,
        )
        self.registered_callbacks.append(cfg)

    def execute_callbacks(
        self,
    ) -> None:
        if not self.registered_callbacks:
            return

        if len(self.registered_callbacks) > 1:
            raise RuntimeError(
                "execute_callbacks currently supports exactly one registered "
                "callback. Multiple-callback execution will be implemented later."
            )

        cfg = self.registered_callbacks[0]
        simulation_callback = cfg.simulation_callback
        arrays_description = cfg.arrays_description
        max_iterations = cfg.max_iterations
        prepare_iteration = cfg.prepare_iteration
        preparation_advance = cfg.preparation_advance

        max_pending_arrays = 2 * len(arrays_description)

        # Convert the definitions to the type expected by the head node
        head_arrays_description = [(definition.name, definition.preprocess) for definition in arrays_description ]
    
        # TODO maybe this goes in the register callbacks
        ray.get(self.head.register_arrays.remote(head_arrays_description, max_pending_arrays))
    
        arrays_by_iteration: dict[int, dict[str, da.Array]] = {}
    
        if prepare_iteration is not None:
            preparation_results: dict[int, ray.ObjectRef] = {}
    
            for timestep in range(min(preparation_advance, max_iterations)):
                # Get the next array from the head node
                array: da.Array = ray.get(self.head.get_preparation_array.remote(arrays_description[0].name, timestep))
                preparation_results[timestep] = _call_prepare_iteration.remote(prepare_iteration, array, timestep)
    
        for iteration in range(max_iterations):
            # Start preparing in advance
            if iteration + preparation_advance < max_iterations and prepare_iteration is not None:
                array = self.head.get_preparation_array.remote(arrays_description[0].name, iteration + preparation_advance)
                preparation_results[iteration + preparation_advance] = _call_prepare_iteration.remote(
                    prepare_iteration, array, iteration + preparation_advance
                )
    
            # Get new arrays
            while len(arrays_by_iteration.get(iteration, {})) < len(arrays_description):
                name: str
                timestep: int
                array: da.Array
                name, timestep, array = ray.get(self.head.get_next_array.remote())
    
                if timestep not in arrays_by_iteration:
                    arrays_by_iteration[timestep] = {}
    
                assert name not in arrays_by_iteration[timestep]
                arrays_by_iteration[timestep][name] = array
    
            # Compute the arrays to pass to the callback
            all_arrays: dict[str, da.Array | list[da.Array]] = {}
    
            for description in arrays_description:
                if description.window_size is None:
                    all_arrays[description.name] = arrays_by_iteration[iteration][description.name]
                else:
                    all_arrays[description.name] = [
                        arrays_by_iteration[timestep][description.name]
                        for timestep in range(max(iteration - description.window_size + 1, 0), iteration + 1)
                    ]
    
            if prepare_iteration is not None:
                preparation_result = ray.get(preparation_results[iteration])
                simulation_callback(**all_arrays, timestep=timestep, preparation_result=preparation_result)
            else:
                simulation_callback(**all_arrays, timestep=timestep)
    
            del all_arrays
    
            # Remove the oldest arrays
            for description in arrays_description:
                older_timestep = iteration - (description.window_size or 1) + 1
                if older_timestep >= 0:
                    del arrays_by_iteration[older_timestep][description.name]
    
                    if not arrays_by_iteration[older_timestep]:
                        del arrays_by_iteration[older_timestep]
    
            # Free the memory used by the arrays now. Since an ObjectRef is a small object,
            # Python may otherwise choose to keep it in memory for some time, preventing the
            # actual data to be freed.
            gc.collect()
