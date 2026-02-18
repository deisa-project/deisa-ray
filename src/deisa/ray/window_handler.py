from collections import deque, defaultdict
import gc
import logging
from typing import Any, Callable, Hashable, List, Optional, Literal

import dask
from deisa.core.interface import SupportsSlidingWindow
import ray
from ray.util.dask import ray_dask_get

from deisa.ray._scheduler import deisa_ray_get
from deisa.ray.config import config
from deisa.ray.errors import _default_exception_handler
from deisa.ray.head_node import HeadNodeActor
from deisa.ray.types import (
    ActorID,
    DeisaArray,
    RayActorHandle,
    WindowSpec,
    _CallbackConfig,
)
from deisa.ray.utils import get_head_actor_options


def _ray_start_impl() -> None:
    """
    Default Ray startup procedure used by :class:`Deisa`.

    Notes
    -----
    Initializes Ray only once with minimal logging. Used when the caller
    does not provide a custom ``ray_start`` hook.
    """
    if not ray.is_initialized():
        ray.init(address="auto", log_to_driver=False, logging_level=logging.ERROR)


class Deisa:
    """
    Entry point that orchestrates analytics callbacks on Ray.

    Provides an API for registering sliding window callbacks and executing
    them as arrays arrive from simulation ranks.
    """

    def __init__(
        self,
        *,
        ray_start: Optional[Callable[[], None]] = None,
        max_simulation_ahead: int = 1,
    ) -> None:
        """
        Initialize handler state without touching Ray.

        Parameters
        ----------
        ray_start : Callable[[], None], optional
            Custom callable used to start Ray. Defaults to a built-in helper.
        max_simulation_ahead : int, optional
            Number of timesteps the analytics may lag behind the simulation.
            Defaults to 1.
        """
        # cheap constructor: no Ray side effects
        config.lock()
        self._experimental_distributed_scheduling_enabled = config.experimental_distributed_scheduling_enabled

        # Do NOT mutate global config here if you want cheap unit tests;
        # do it when connecting, or inject it similarly.
        self._ray_start = ray_start or _ray_start_impl

        self._connected = False
        self.node_actors: dict[ActorID, RayActorHandle] = {}
        self.registered_callbacks: list[_CallbackConfig] = []
        self.queue_per_array: dict[str, deque]
        self.max_simulation_ahead: int = max_simulation_ahead
        self.has_new_timestep: dict[str, bool] = defaultdict(bool)
        self.has_seen_array: dict[str, bool] = defaultdict(bool)
        self.queue_per_array = {}

    def _ensure_connected(self) -> None:
        """
        Ensure the handler is connected to Ray and has a head actor ready.

        Notes
        -----
        Starts Ray (if needed), configures Dask to use the correct scheduler,
        creates the head actor, and exchanges configuration so that
        scheduling actors can register themselves.
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
        # readiness gate for head actor - only return when its alive
        ray.get(
            self.head.exchange_config.remote(
                {"experimental_distributed_scheduling_enabled": self._experimental_distributed_scheduling_enabled}
            )
        )
        self._connected = True

    def _create_head_actor(self) -> None:
        """
        Instantiate the head actor that coordinates array delivery.

        Notes
        -----
        Uses :func:`get_head_actor_options` to pin the actor to the Ray head node
        with a detached lifetime so that analytics can connect later.
        """
        self.head = HeadNodeActor.options(**get_head_actor_options()).remote(
            max_simulation_ahead=self.max_simulation_ahead
        )

    def callback(
        self,
        *window_specs,
        exception_handler: Optional[SupportsSlidingWindow.ExceptionHandler] = None,
        when: Literal["AND", "OR"] = "AND",
    ):
        """
        Decorator that registers a sliding-window analytics callback.

        Parameters
        ----------
        *window_specs : WindowSpec
            Array descriptions the callback should receive.
        exception_handler : Optional[SupportsSlidingWindow.ExceptionHandler], optional
            Handler invoked when the user callback raises. Defaults to
            :func:`deisa.ray.errors._default_exception_handler`.
        when : Literal["AND", "OR"], optional
            Governs whether all arrays (``"AND"``) or any array (``"OR"``)
            must be available before the callback runs. Defaults to ``"AND"``.

        Returns
        -------
        Callable[[SupportsSlidingWindow.Callback], SupportsSlidingWindow.Callback]
            Decorator that registers ``simulation_callback`` with the window
            handler.
        """

        def deco(fn):
            return self.register_callback(fn, list(window_specs), exception_handler, when)

        return deco

    def register_callback(
        self,
        simulation_callback: SupportsSlidingWindow.Callback,
        arrays_spec: list[WindowSpec],
        exception_handler: Optional[SupportsSlidingWindow.ExceptionHandler] = None,
        when: Literal["AND", "OR"] = "AND",
    ) -> SupportsSlidingWindow.Callback:
        """
        Register the analytics callback and array descriptions.

        Parameters
        ----------
        simulation_callback : SupportsSlidingWindow.Callback
            Function to run for each iteration; receives arrays as kwargs
            and ``timestep``.
        arrays_spec : list[WindowSpec]
            Descriptions of arrays to stream to the callback (with optional
            sliding windows).
            Maximum iterations to execute. Default is a large sentinel.
        exception_handler : Optional[SupportsSlidingWindow.ExceptionHandler]
            Exception handler to handle any exception thrown by simulation
            (like division by zero). Defaults to printing the error and moving on.
        when : Literal['AND', 'OR']
            When callback have multiple arrays, govern when callback should be called.
            `AND`: only call callback if ALL required arrays have been shared for a given timestep.
            `OR`: call callback if ANY array has been shared for a given timestep.

        Returns
        -------
        SupportsSlidingWindow.Callback
            The original callback, allowing decorator-style usage.
        """
        self._ensure_connected()  # connect + handshake before accepting callbacks
        cfg = _CallbackConfig(
            simulation_callback=simulation_callback,
            arrays_description=arrays_spec,
            exception_handler=exception_handler or _default_exception_handler,
            when=when,
        )
        self.registered_callbacks.append(cfg)
        return simulation_callback

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
        """
        Prepare per-array queues that respect declared window sizes.

        Notes
        -----
        Each queue is a :class:`collections.deque` with ``maxlen`` matching the
        largest window requested for that array.
        """
        for cb_cfg in self.registered_callbacks:
            description = cb_cfg.arrays_description
            for array_def in description:
                name = array_def.name
                window_size: int = array_def.window_size if array_def.window_size is not None else 1

                if name in self.queue_per_array:
                    if self.queue_per_array[name].maxlen < window_size:
                        self.queue_per_array[name] = deque(maxlen=window_size)
                    else:
                        pass
                else:
                    self.queue_per_array[name] = deque(maxlen=window_size)

    def execute_callbacks(
        self,
    ) -> None:
        """
        Execute the registered simulation callback loop.

        Notes
        -----
        Supports a single registered callback at present. Manages array
        retrieval from the head actor, windowed
        array delivery, and garbage collection between iterations.
        """
        # ensure connected to ray cluster
        self._ensure_connected()

        # signal analytics ready to start
        ray.get(self.head.set_analytics_ready_for_execution.remote())
        # ray.get(self.head.wait_for_bridges_ready.remote())

        # TODO: test
        # raise error and kill analytics
        if not self.registered_callbacks:
            raise RuntimeError("Please register at least one callback before calling execute_callbacks()")

        # generate one queue per array which cleanly handles the window size
        self.generate_queue_per_array()

        # get first array to kickstart the process
        # - Add to queue, mark as new timestep arrived
        name, arr_timestep, array = ray.get(self.head.get_next_array.remote())
        if name == "__deisa_last_iteration_array":
            return

        queue = self.queue_per_array.get(name)
        if queue is not None:
            queue.append(DeisaArray(dask=array, t=arr_timestep))
            self.has_new_timestep[name] = True
            self.has_seen_array[name] = True

        end_reached = False
        while not end_reached:
            # inner while loop stops once a bigger timestep has been pushed to queue
            # WARNING: Big assumption is that it is impossible for any array in timestep i+1 to be placed
            # BEFORE timestep i. This is violated in embarrassingly parallel workflows where each rank can go ahead
            # independently. Without this assumption, it would be much more complex to determine a good moment to analyze
            # which callbacks should be called - as such, memory handling and flow execution become difficult to
            # guarantee.
            current_timestep = arr_timestep
            while True:
                name, arr_timestep, array = ray.get(self.head.get_next_array.remote())
                # guarantee sequential flow of data.
                # TODO add test
                if arr_timestep < current_timestep:
                    raise RuntimeError(
                        f"Logical flow of data was violated. Timestep {arr_timestep} sent after timestep {current_timestep}. Exiting..."
                    )
                if name == "__deisa_last_iteration_array":
                    end_reached = True
                    break
                # simulation has produced a higher timestep -> process all arrays for current_timestep
                if arr_timestep > current_timestep:
                    break

                queue = self.queue_per_array.get(name)
                if queue is not None:
                    queue.append(DeisaArray(dask=array, t=arr_timestep))
                    self.has_new_timestep[name] = True
                    self.has_seen_array[name] = True

            # inspect what callbacks can be called
            for cb_cfg in self.registered_callbacks:
                simulation_callback = cb_cfg.simulation_callback
                description_arrays_needed = cb_cfg.arrays_description
                exception_handler = cb_cfg.exception_handler
                when = cb_cfg.when

                should_call = self.should_call(description_arrays_needed, when)
                if should_call:
                    # Compute the arrays to pass to the callback
                    callback_args: dict[str, List[DeisaArray]] = self.determine_callback_args(description_arrays_needed)
                    try:
                        simulation_callback(**callback_args)
                    except TimeoutError as e:
                        raise e
                    except AssertionError as e:
                        raise e
                    except BaseException as e:
                        try:
                            exception_handler(e)
                        except BaseException as e:
                            _default_exception_handler(e)

                    del callback_args
                    gc.collect()

            # set all new timesteps to be false
            for queue in self.has_new_timestep:
                self.has_new_timestep[queue] = False

            # add the first "bigger" timestep back into queue and set new_timestep flag
            if not end_reached:
                queue = self.queue_per_array.get(name)
                if queue is not None:
                    queue.append(DeisaArray(dask=array, t=arr_timestep))
                    self.has_new_timestep[name] = True
                    self.has_seen_array[name] = True

    def determine_callback_args(self, description_of_arrays_needed) -> dict[str, List[DeisaArray]]:
        """
        Build the kwargs passed to a simulation callback.

        Parameters
        ----------
        description_of_arrays_needed : Sequence[WindowSpec]
            Array descriptions requested by the callback.

        Returns
        -------
        dict[str, List[DeisaArray]]
            Mapping from array name to the latest (windowed) list of ``DeisaArray`` instances.
        """
        callback_args = {}
        for description in description_of_arrays_needed:
            name = description.name
            window_size = description.window_size
            queue = self.queue_per_array[name]
            if window_size is None:
                callback_args[name] = [queue[-1]]
            else:
                callback_args[name] = list(queue)[-window_size:]
        return callback_args

    def should_call(self, description_of_arrays_needed, when: Literal["AND", "OR"]) -> bool:
        """
        Determine whether a callback should execute for the current state.

        Parameters
        ----------
        description_of_arrays_needed : Sequence[WindowSpec]
            Array descriptions governing the callback.
        when : Literal["AND", "OR"]
            Execution mode specifying whether all arrays or any array must have
            new data.

        Returns
        -------
        bool
            ``True`` when the callback criteria are met.
        """
        names = [d.name for d in description_of_arrays_needed]
        if when == "AND":
            return all(self.has_new_timestep[n] for n in names)
        else:  # when == 'OR'
            return all(self.has_seen_array[n] for n in names) and any(self.has_new_timestep[n] for n in names)

    # TODO add persist
    def set(self, *, key: Hashable, value: Any, chunked: bool = False, persist: bool = False) -> None:
        """
        Broadcast a feedback value to all scheduling actors.

        Parameters
        ----------
        key : Hashable
            Identifier for the shared value.
        value : Any
            Value to distribute.
        chunked : bool, optional
            Placeholder for future distributed-array feedback. Only ``False``
            is supported today. Default is ``False``.
        persist : bool, optional
            Whether the value should survive the next retrieval.
            Defaults to ``False``.

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
                handle.set.remote(key, value, chunked, persist)
        else:
            # TODO: implement chunked version
            raise NotImplementedError()
