import gc
from dataclasses import dataclass
from typing import Any, Callable

import dask
import dask.array as da
import ray

from deisa.ray._scheduler import deisa_ray_get
from deisa.ray.head_node import ArrayDefinition as HeadArrayDefinition
from deisa.ray.head_node import SimulationHead, get_head_actor_options


@dataclass
class ArrayDefinition:
    """
    Description of an array with optional windowing support.

    Parameters
    ----------
    name : str
        The name of the array.
    window_size : int or None, optional
        If specified, creates a sliding window of arrays for this array name.
        The window will contain the last `window_size` timesteps. If None,
        only the current timestep array is provided. Default is None.
    preprocess : Callable, optional
        A preprocessing function to apply to chunks of this array before
        they are sent to the analytics. The function should take a numpy
        array and return a processed numpy array. Default is the identity
        function (no preprocessing).

    Examples
    --------
    >>> def normalize(arr):
    ...     return arr / arr.max()
    >>> # Array with windowing: last 5 timesteps
    >>> array_def = ArrayDefinition(name="temperature", window_size=5, preprocess=normalize)
    >>> # Array without windowing: current timestep only
    >>> array_def = ArrayDefinition(name="pressure", window_size=None)
    """

    name: str
    window_size: int | None = None
    preprocess: Callable = lambda x: x


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


def run_simulation(
    simulation_callback: Callable,
    arrays_description: list[ArrayDefinition],
    *,
    max_iterations=1000_000_000,
    prepare_iteration: Callable | None = None,
    preparation_advance: int = 3,
) -> None:
    """
    Run a simulation that processes arrays from the Ray cluster with optional windowing.

    This function coordinates the execution of a simulation callback that processes
    arrays received from the Ray cluster. It supports sliding windows for arrays,
    allowing the callback to access arrays from multiple timesteps. The function
    manages array collection, windowing, memory cleanup, and optional preparation
    callbacks.

    Parameters
    ----------
    simulation_callback : Callable
        The main simulation callback function. It will be called with keyword
        arguments for each array (by name) and a `timestep` argument. If
        `prepare_iteration` is provided, it will also receive a
        `preparation_result` argument.
    arrays_description : list[ArrayDefinition]
        List of array definitions describing the arrays to be processed. Each
        definition can specify a window size for sliding window access.
    max_iterations : int, optional
        Maximum number of iterations to run. Default is 1_000_000_000.
    prepare_iteration : Callable or None, optional
        Optional callback function that is called in advance for each timestep
        to prepare data. The function receives a Dask array and timestep. The
        result is passed to the simulation_callback as `preparation_result`.
        Default is None.
    preparation_advance : int, optional
        Number of timesteps ahead to prepare. The prepare_iteration callback
        will be called this many timesteps in advance. Default is 3.

    Notes
    -----
    The function performs the following operations for each iteration:

    1. Collects arrays from the head node until all required arrays for the
       current iteration are available.
    2. Constructs the arrays dictionary, applying windowing for arrays with
       `window_size` specified.
    3. Calls the simulation_callback with the arrays and timestep.
    4. Cleans up old arrays that are no longer needed for the window.
    5. Triggers garbage collection to free memory.

    For arrays with `window_size` set to `n`, the callback receives a list
    of arrays containing the last `n` timesteps. For arrays without windowing,
    the callback receives a single array for the current timestep.

    The function manages memory by deleting arrays that are outside the window
    and explicitly calling garbage collection to ensure Ray object references
    are released promptly.

    Examples
    --------
    >>> def process_data(temperature, pressure, timestep):
    ...     # Process arrays for current timestep
    ...     result = temperature + pressure
    ...     return result
    >>>
    >>> arrays = [
    ...     ArrayDefinition(name="temperature", window_size=5),  # Last 5 timesteps
    ...     ArrayDefinition(name="pressure"),  # Current timestep only
    ... ]
    >>> run_simulation(process_data, arrays, max_iterations=100)
    """
    # Convert the definitions to the type expected by the head node
    head_arrays_description = [
        HeadArrayDefinition(name=definition.name, preprocess=definition.preprocess) for definition in arrays_description
    ]

    # Limit the advance the simulation can have over the analytics
    max_pending_arrays = 2 * len(arrays_description)

    head: Any = SimulationHead.options(**get_head_actor_options()).remote(head_arrays_description, max_pending_arrays)

    arrays_by_iteration: dict[int, dict[str, da.Array]] = {}

    if prepare_iteration is not None:
        preparation_results: dict[int, ray.ObjectRef] = {}

        for timestep in range(min(preparation_advance, max_iterations)):
            # Get the next array from the head node
            array: da.Array = ray.get(head.get_preparation_array.remote(arrays_description[0].name, timestep))
            preparation_results[timestep] = _call_prepare_iteration.remote(prepare_iteration, array, timestep)

    for iteration in range(max_iterations):
        # Start preparing in advance
        if iteration + preparation_advance < max_iterations and prepare_iteration is not None:
            array = head.get_preparation_array.remote(arrays_description[0].name, iteration + preparation_advance)
            preparation_results[iteration + preparation_advance] = _call_prepare_iteration.remote(
                prepare_iteration, array, iteration + preparation_advance
            )

        # Get new arrays
        while len(arrays_by_iteration.get(iteration, {})) < len(arrays_description):
            name: str
            timestep: int
            array: da.Array
            name, timestep, array = ray.get(head.get_next_array.remote())

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
