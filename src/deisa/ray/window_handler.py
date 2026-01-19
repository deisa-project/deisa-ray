import gc
from pydoc import describe
from re import A
from typing import Any, Callable, Hashable, Optional, Literal, List
from collections import defaultdict, deque

import dask
import dask.array as da
from ray.util.dask import ray_dask_get
import ray

from deisa.ray._scheduler import deisa_ray_get
from deisa.ray.head_node import HeadNodeActor
from deisa.ray.utils import get_head_actor_options
from deisa.ray.types import ActorID, RayActorHandle, WindowArrayDefinition, _CallbackConfig, DeisaArray
from deisa.ray.config import config
import logging
from deisa.core.interface import SupportsSlidingWindow
from deisa.ray.errors import _default_exception_handler


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
    def __init__(
        self,
        *,
        max_iterations: int = 1000_000_000,
        ray_start: Optional[Callable[[], None]] = None,
        handshake: Optional[Callable[["Deisa"], None]] = None,
    ) -> None:
        # cheap constructor: no Ray side effects
        config.lock()

        self._experimental_distributed_scheduling_enabled = config.experimental_distributed_scheduling_enabled

        # Do NOT mutate global config here if you want cheap unit tests;
        # do it when connecting, or inject it similarly.
        self._ray_start = ray_start or self._ray_start_impl
        self._handshake = handshake or self._handshake_impl

        self._connected = False
        self.node_actors: dict[ActorID, RayActorHandle] = {}
        self.registered_callbacks: list[_CallbackConfig] = []
        self.max_iterations = max_iterations
        self.queue_per_array: dict[str, deque]

    def _handshake_impl(self, _: "Deisa") -> None:
        """
        Implementation for handshake between window handler (Deisa) and the Simulation side Bridges.

        The handshake occurs when all the expected Ray Node Actors are connected.

        :param self: Description
        :param _: Description
        :type _: "Deisa"
        """
        # TODO finish and add this config option to Deisa
        self.total_nodes = 0
        from ray.util.state import list_actors

        expected_ray_actors = self.total_nodes
        connected_actors = 0
        while connected_actors < expected_ray_actors:
            connected_actors = 0
            for a in list_actors(filters=[("state", "=", "ALIVE")]):
                if a.get("ray_namespace") == "deisa_ray":
                    connected_actors += 1

    def _ensure_connected(self) -> None:
        """
        Ensures that the widow handler has connected to the Ray Cluster.

        This function connects to ray, creates a head_actor, and waits until a
        handshake occurs, which happens when all node actors have connected to
        the cluster. It also changes the dask on ray scheduler based on whether the
        user wants to set centralized scheduling or not.

        :param self: Description
        """
        if self._connected:
            return

        # Side effects begin here (only once)
        self._ray_start()

        # configure dask here if it must reflect actual cluster runtime
        if self._experimental_distributed_scheduling_enabled:
            dask.config.set(scheduler=deisa_ray_get, shuffle="tasks")
        else:
            dask.config.set(scheduler=ray_dask_get, shuffle="tasks")

        # head is created
        self._create_head_actor()
        # readyness gate for head actor - only return when its alive
        ray.get(
            self.head.exchange_config.remote(
                {"experimental_distributed_scheduling_enabled": self._experimental_distributed_scheduling_enabled}
            )
        )

        self._handshake(self)

        self._connected = True

    def _create_head_actor(self) -> None:
        self.head = HeadNodeActor.options(**get_head_actor_options()).remote()

    def _ray_start_impl(self) -> None:
        if not ray.is_initialized():
            ray.init(address="auto", log_to_driver=False, logging_level=logging.ERROR)

    def register_callback(
        self,
        simulation_callback: SupportsSlidingWindow.Callback,
        arrays_description: list[WindowArrayDefinition],
        exception_handler: Optional[SupportsSlidingWindow.ExceptionHandler] = None,
        when: Literal["AND", "OR"] = "AND",
        # *,
        # prepare_iteration: Callable | None = None,
        # preparation_advance: int = 3,
    ) -> None:
        """
        Register the analytics callback and array descriptions.

        Parameters
        ----------
        simulation_callback : Callable
            Function to run for each iteration; receives arrays as kwargs
            and ``timestep``.
        arrays_description : list[WindowArrayDefinition]
            Descriptions of arrays to stream to the callback (with optional
            sliding windows).
        max_iterations : int, optional
            Maximum iterations to execute. Default is a large sentinel.
        prepare_iteration : DEPRECATED Callable or None, optional
            Optional preparatory callback run ``preparation_advance`` steps
            ahead. Receives the array and ``timestep``.
        preparation_advance : DEPRECATED int, optional
            How many iterations ahead to prepare when ``prepare_iteration``
            is provided. Default is 3.
        """
        self._ensure_connected()  # connect + handshake before accepting callbacks
        cfg = _CallbackConfig(
            simulation_callback=simulation_callback,
            arrays_description=arrays_description,
            exception_handler=exception_handler or _default_exception_handler,
            # prepare_iteration=prepare_iteration,
            # preparation_advance=preparation_advance,
        )
        self.registered_callbacks.append(cfg)

        # register array with head actor
        max_pending_arrays = 2 * len(arrays_description)

        # TODO make head node take the entire type
        head_arrays_description = [(definition.name, definition.preprocess) for definition in arrays_description]

        ray.get(self.head.register_arrays.remote(head_arrays_description, max_pending_arrays))

    def unregister_callback(
        self,
        simulation_callback: Callable,
    ) -> None:
        """
        Unregister a previously registered simulation callback.

        Parameters
        ----------
        simulation_callback : Callable
            Callback to remove from the registry.

        Raises
        ------
        NotImplementedError
            Always, as the feature has not been implemented yet.
        """
        raise NotImplementedError("method not yet implemented.")

    def generate_queue_per_array(self):
        self.queue_per_array = {}
        for cb_cfg in self.registered_callbacks:
            description = cb_cfg.arrays_description
            for arraydef in description:
                name = arraydef.name
                window_size = arraydef.window_size

                if name in self.queue_per_array:
                    if self.queue_per_array[name].maxlen < window_size:
                        self.queue_per_array[name] = deque(maxlen=window_size)
                    else:
                        pass
                else:
                    self.queue_per_array[name] = deque(maxlen=window_size)

    # TODO: introduce a method that will generate the final array spec for each registered array
    def execute_callbacks(
        self,
    ) -> None:
        """
        Execute the registered simulation callback loop.

        Notes
        -----
        Supports a single registered callback at present. Manages array
        retrieval from the head actor, optional preparation tasks, windowed
        array delivery, and garbage collection between iterations.
        """
        self._ensure_connected()
        head_arrays_description = [("__deisa_last_iteration_array", None)]
        ray.get(self.head.register_arrays.remote(head_arrays_description, 0))
        ray.get(self.head.set_semaphore.remote())

        if not self.registered_callbacks:
            raise RuntimeError("Please register at least one callback before calling execute_callbacks()")

        self.generate_queue_per_array()

        # TODO sim should signal end
        name, timestep, array = ray.get(self.head.get_next_array.remote())
        self.queue_per_array[name].append(DeisaArray(dask=array, t=timestep))
        end_reached = False
        while not end_reached:
            # get next available array
            time = timestep
            while True:
                name, timestep, array = ray.get(self.head.get_next_array.remote())
                if name == "__deisa_last_iteration_array":
                    end_reached = True
                    break
                if time < timestep:
                    break

                self.queue_per_array[name].append(DeisaArray(dask=array, t=timestep))

            # inspect what callbacks can be called
            for cb_cfg in self.registered_callbacks:
                simulation_callback = cb_cfg.simulation_callback
                description_arrays_needed = cb_cfg.arrays_description
                exception_handler = cb_cfg.exception_handler

                # Compute the arrays to pass to the callback
                callback_args: dict[str, DeisaArray | List[DeisaArray]] = self.determine_callback_args(
                    description_arrays_needed
                )
                try:
                    simulation_callback(**callback_args)
                except TimeoutError:
                    pass
                except BaseException as e:
                    try:
                        exception_handler(e)
                    except BaseException as e:
                        _default_exception_handler(e)

                del callback_args
                gc.collect()
            if not end_reached:
                self.queue_per_array[name].append(DeisaArray(dask=array, t=timestep))

    def determine_callback_args(self, description_of_arrays_needed) -> dict[str, List[DeisaArray]]:
        callback_args: dict[str, list[DeisaArray]] = {}
        for description in description_of_arrays_needed:
            name = description.name
            window_size = description.window_size
            queue = self.queue_per_array[name]
            if window_size is None:
                callback_args[name] = [queue[-1]]
            else:
                callback_args[name] = list(queue)[-window_size:]
        for name, arrays in callback_args.items():
            if len(callback_args[name]) == 1:
                callback_args[name] = arrays[0]
        return callback_args

    # TODO add persist
    def set(self, *args, key: Hashable, value: Any, chunked: bool = False, **kwargs) -> None:
        """
        Broadcast a feedback value to all node actors.

        Parameters
        ----------
        key : Hashable
            Identifier for the shared value.
        value : Any
            Value to distribute.
        chunked : bool, optional
            Placeholder for future distributed-array feedback. Only ``False``
            is supported today. Default is ``False``.

        Notes
        -----
        The method lazily fetches node actors once and uses fire-and-forget
        remote calls; callers should not assume synchronous delivery.
        """
        # TODO test
        if not self.node_actors:
            # retrieve node actors at least once
            self.node_actors = ray.get(self.head.list_scheduling_actors.remote())

        if not chunked:
            for _, handle in self.node_actors.items():
                # set the value inside each node actor
                # TODO: does it need to be blocking?
                handle.set.remote(key, value, chunked)
        else:
            # TODO: implement chunked version
            raise NotImplementedError()
