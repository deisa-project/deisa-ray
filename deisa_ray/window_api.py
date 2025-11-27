import gc
from dataclasses import dataclass
from typing import Any, Callable

import dask
import dask.array as da
import ray

from doreisa._scheduler import doreisa_get
from doreisa.head_node import ArrayDefinition as HeadArrayDefinition
from doreisa.head_node import SimulationHead, get_head_actor_options


@dataclass
class ArrayDefinition:
    name: str
    window_size: int | None = None
    preprocess: Callable = lambda x: x


@ray.remote(num_cpus=0, max_retries=0)
def _call_prepare_iteration(prepare_iteration: Callable, array: da.Array, timestep: int):
    """
    Call the prepare_iteration function with the given array and timestep.

    Args:
        prepare_iteration: The function to call.
        array: The Dask array to pass to the function.
        timestep: The current timestep.

    Returns:
        The result of the prepare_iteration function.
    """
    dask.config.set(scheduler=doreisa_get, shuffle="tasks")
    return prepare_iteration(array, timestep=timestep)


def run_simulation(
    simulation_callback: Callable,
    arrays_description: list[ArrayDefinition],
    *,
    max_iterations=1000_000_000,
    prepare_iteration: Callable | None = None,
    preparation_advance: int = 3,
) -> None:
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
