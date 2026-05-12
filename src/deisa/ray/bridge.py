"""Bridge between MPI ranks and the Ray-based analytics system.

This module exposes the :class:`Bridge` class used by simulation ranks to
register their data chunks and exchange information with analytics running on
top of Ray.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Mapping
import numpy as np
import ray
from ray.actor import ActorClass
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from deisa.core import ICommunicator
from deisa.ray import Timestep
from deisa.ray.comm import normalize_comm
from deisa.ray.errors import ContractError, _default_exception_handler
from deisa.ray.scheduling_actor import SchedulingActor as _RealSchedulingActor
from deisa.ray.types import RayActorHandle
from deisa.ray.validate import _validate_arrays_meta
from deisa.ray.utils import get_node_actor_options
import sys

logger = logging.getLogger(__name__)


class Bridge:
    """
    Bridge between MPI ranks and Ray cluster for distributed array processing.

    Each Bridge instance is created by an MPI rank to connect to the Ray cluster
    and send data chunks. Each Bridge is responsible for managing a chunk of data
    from the decomposed distributed array.

    Parameters
    ----------
    arrays_metadata : Mapping[str, Mapping[str, Any]]
        Metadata describing the array layout managed by this bridge.
    comm : deisa.core.ICommunicator
        Communication backend for the simulation ranks. The bridge ID is
        derived from ``comm.Get_rank()``. Raw ``mpi4py`` communicators are
        wrapped in :class:`deisa.ray.comm.MPICommAdapter`.
    _node_id : str or None, optional
        Node identifier used for testing or custom scheduling. Defaults to ``None``.
    scheduling_actor_cls : Type, optional
        Class used to materialize the scheduling actor. Defaults to
        :class:`deisa.ray.scheduling_actor.SchedulingActor`.
    _init_retries : int, optional
        Number of attempts to create and ready the node actor. Defaults to 3.

    Attributes
    ----------
    node_id : str
        The ID of the node this Bridge is associated with.

    Notes
    -----
    The Bridge automatically initializes Ray if it hasn't been initialized yet.
    The scheduling actor is created with a detached lifetime to persist beyond
    the Bridge initialization. The actor uses node affinity scheduling to ensure
    it runs on the specified node.

    Examples
    --------
    Create a bridge for a simulation rank that owns one array ``temperature``::

        arrays_metadata = {
            "temperature": {
                "global_shape": (40, 40),
                "chunk_shape": (10, 10),
                "chunk_position": (0, 0),
            }
        }
        bridge = Bridge(
            arrays_metadata=arrays_metadata,
            comm=comm,
        )

        bridge.send(
            array_name="temperature",
            chunk=np.zeros((10, 10), dtype=np.float64),
            timestep=0
        )
    """

    # TODO: add exception handler? what should default be? If bridge is not instantiated,
    # should sim crash? Keep going?
    def __init__(
        self,
        arrays_metadata: Mapping[str, Mapping[str, Any]],
        comm: ICommunicator,
        *,
        _node_id: str | None = None,
        scheduling_actor_cls: ActorClass = _RealSchedulingActor,
        _init_retries: int = 3,
    ) -> None:
        """
        Initialize the Bridge to connect MPI rank to Ray cluster.

        Parameters
        ----------
        arrays_metadata : Mapping[str, Mapping[str, Any]]
            Dictionary that describes the arrays being shared by the simulation.
            Keys represent the name of the array while the values are
            dictionaries that must at least declare the metadata expected by
            :meth:`validate_arrays_meta`.
        comm : deisa.core.ICommunicator
            Communication backend to use. The unique bridge identifier is
            derived from ``comm.Get_rank()``. Raw ``mpi4py`` communicators are
            wrapped in :class:`deisa.ray.comm.MPICommAdapter`.
        _node_id : str or None, optional
            The ID of the node. If None, the ID is taken from the Ray runtime
            context. Useful for testing with several scheduling actors on a
            single machine. Default is None.
        scheduling_actor_cls : Type, optional
            The class to use for creating the scheduling actor. Default is
            `_RealSchedulingActor`.
        _init_retries : int, optional
            Number of retry attempts when initializing the scheduling actor.
            Default is 3.
        Raises
        ------
        RuntimeError
            If the scheduling actor cannot be created or initialized after
            the specified number of retries.
        ValueError
            If ``comm`` is ``None``.
        TypeError
            If ``comm`` does not implement :class:`deisa.core.ICommunicator`.

        Notes
        -----
        This method automatically initializes Ray if it hasn't been initialized
        yet. The scheduling actor is created with a detached lifetime and uses
        node affinity scheduling when `_node_id` is None. The first remote call
        to the scheduling actor serves as a readiness check.
        """
        self._init_retries = _init_retries
        self._closed = False

        self.arrays_metadata = _validate_arrays_meta(arrays_metadata)
        comm = normalize_comm(comm)
        if comm is None:
            raise ValueError("comm is required")
        if not isinstance(comm, ICommunicator):
            raise TypeError("comm must implement deisa.core.ICommunicator")
        self.comm = comm
        self.bridge_id = self.comm.Get_rank()

        # NOTE : Possible error : if two bridges have different first array it will have different
        # shape and will be declared twice.
        # Possible fix : do it somewhere else (Head or SchedulingActor)
        # we add a special array with a name that will signal the end of the simulation
        # note we only need the metadata so that it can pass through the entire pipeline correctly and
        # in sequential order, so we just replicate the first metadata we have.

        if self.bridge_id == 0:
            self.arrays_metadata["__deisa_last_iteration_array"] = {
                "global_shape": (1, 1),
                "chunk_shape": (1, 1),
                "chunk_position": (0, 0),
            }

        if not ray.is_initialized():
            ray.init(address="auto", log_to_driver=False, logging_level=logging.ERROR)

        self.node_id = _node_id or ray.get_runtime_context().get_node_id()
        name = f"sched-{self.node_id}"
        namespace = "deisa_ray"

        node_actor_options: Dict[str, Any] = get_node_actor_options(name, namespace)

        # place node actor in the same place as detected node_id
        if _node_id is None:
            node_actor_options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
                node_id=self.node_id,
                soft=False,
            )

        # create node actor
        self._create_node_actor(scheduling_actor_cls, node_actor_options)
        self.head_actor: RayActorHandle | None = None
        # exchange meta with node actor (blocking call)
        self._exchange_chunks_meta_with_node_actor()
        # make sure node actor is ready
        ray.get(self.node_actor.ready.remote())

        # barrier to make sure that all ranks have reached this point
        self.comm.barrier()

        # after this function returns we are sure that
        # 1. analytics have started
        # 2. head actor is created
        # 3. all bridges have connected
        # 4. all node actors have been created
        # 5. all node actors have received description of arrays_md

        # ray method calls are sequential on same actor
        ray.get(self.node_actor.finalize_registration.remote())


    def _exchange_chunks_meta_with_node_actor(self):
        """
        Push per-array metadata to the node actor.

        Notes
        -----
        This method registers both global chunk layout and the bridge-specific
        chunk position for every array described in ``arrays_metadata``.
        """
        # send metadata of each array chunk (both global and local chunk info) to node actor

        refs = []
        for array_name, meta in self.arrays_metadata.items():
            refs.append(
                self.node_actor.register_chunk_meta.remote(
                    # global info of array (same across bridges)
                    array_name=array_name,
                    chunk_shape=meta["chunk_shape"],
                    global_shape=meta["global_shape"],
                    # local info of array specific to bridge
                    bridge_id=self.bridge_id,
                    chunk_position=meta["chunk_position"],
                )
            )
        ray.get(refs)

    def send(
        self,
        *,
        array_name: str,
        chunk: np.ndarray,
        timestep: int,
        test_mode: bool = False,
    ) -> None:
        """
        Make a chunk of data available to the analytics.

        This method stores the ``chunk`` in Ray's object store, and
        sends a reference to the node actor. The method blocks until the
        data is processed by the node actor.

        Parameters
        ----------
        array_name : str
            The name of the array this chunk belongs to.
        chunk : numpy.ndarray
            The chunk of data to be sent to the analytics.
        timestep : int
            The timestep index for this chunk of data.
        test_mode : bool, optional
            Reserved flag for future testing or validation hooks. Currently
            ignored. Default is False.

        Notes
        -----
        The chunk is stored in Ray's object store with the node actor as the owner,
        ensuring the reference persists even after the simulation script terminates.
        This method blocks until the node actor has the chunk.

        Raises
        ------
        ContractError
            When the scheduling node detects a contract violation for the
            provided chunk.

        Returns
        -------
        None
            Blocks until the node actor processes the chunk.
        """
        try:
            chunk_dtype = chunk.dtype
            # Setting the owner allows keeping the reference when the simulation script terminates.
            ref = ray.put(chunk, _owner=self.node_actor)
            future: ray.ObjectRef = self.node_actor.add_chunk.remote(
                bridge_id=self.bridge_id,
                array_name=array_name,
                chunk_ref=[ref],
                dtype=chunk_dtype,
                timestep=timestep,
            )  # type: ignore
            # Wait until the data is processed before returning to the simulation
            ray.get(future)
        except ContractError as e:
            raise e
        except Exception as e:
            _default_exception_handler(e)

    def __del__(self):
        try:
            if hasattr(self, "comm") and hasattr(self, "node_actor"):
                self.close(sys.maxsize)
        except Exception:
            logger.debug("Ignoring exception while finalizing Bridge", exc_info=True)

    def close(
        self,
        timestep: int,
    ) -> int:
        """
        Close the bridge by signaling analytics that the simulation finished.

        Parameters
        ----------
        timestep : int
            The timestep index corresponding to the sentinel chunk.

        Returns
        -------
        int
            The final timestep after the sentinel chunk is known by the node actor.
        """
        if self._closed:
            return timestep
        self._closed = True

        self.comm.barrier()
        if self.bridge_id == 0:
            try:
                chunk = np.asarray(0)
                chunk_dtype = chunk.dtype
                ref = ray.put(chunk, _owner=self.node_actor)
                future: ray.ObjectRef = self.node_actor.add_chunk.remote(
                    bridge_id=self.bridge_id,
                    array_name="__deisa_last_iteration_array",
                    chunk_ref=[ref],
                    dtype=chunk_dtype,
                    timestep=timestep,
                )  # type: ignore
                ray.get(future)
                logger.info("Bridge %s closed at timestep %s", self.bridge_id, timestep)
            except Exception as e:
                _default_exception_handler(e)
        return timestep

    def _create_node_actor(
        self,
        node_actor_cls: ActorClass = _RealSchedulingActor,
        node_actor_options: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Create (or get) the node actor and register arrays metadata.

        Parameters
        ----------
        node_actor_cls : Type, optional
            Class used to create the node actor. Defaults to
            :class:`deisa.ray.scheduling_actor.SchedulingActor`.
        node_actor_options : Mapping[str, Any] or None, optional
            Options passed to ``node_actor_cls.options(**node_actor_options)``.
            If None, :func:`get_node_actor_options` is used with the default
            name/namespace derived from the bridge.

        Raises
        ------
        RuntimeError
            If the node actor cannot be created or readied after
            ``_init_retries`` attempts.
        """
        if node_actor_options is None:
            node_actor_options = get_node_actor_options(
                name=f"sched-{self.node_id}",
                namespace="deisa_ray",
            )

        last_err = None
        for _ in range(max(1, self._init_retries)):
            try:
                # first rank to arrive creates, others get same handle (get_if_exists)
                self.node_actor: RayActorHandle = node_actor_cls.options(**node_actor_options).remote(
                    actor_id=self.node_id
                )  # type: ignore
                break  # success
            except Exception as e:
                last_err = e
                # Try to re-create a fresh actor instance (same name will resolve to existing or new one)
                continue
        # `else:` clause belongs to for loop. Only executed if it finishes normally without
        #  encountering a `break`.
        else:
            raise RuntimeError(f"Failed to create/ready scheduling actor for node {self.node_id}") from last_err

    def _get_head_actor(self) -> RayActorHandle:
        """
        Return the global head actor handle, caching it on bridge ``0``.

        Notes
        -----
        Bridge ``0`` is the only rank that reads global feedback from Ray.
        Other ranks receive the result through the communicator broadcast in
        :meth:`get`.
        """
        if self.head_actor is None:
            self.head_actor = ray.get_actor(name="simulation_head", namespace="deisa_ray")
            ray.get(self.head_actor.ready.remote())
        return self.head_actor

    def get(
        self,
        key: str,
        timestep: Timestep | None = None,
        default: Any | None = None,
    ) -> Any | None:
        """
        Retrieve feedback from analytics to influence the simulation.

        Bridge ``0`` queries the global head actor directly, then broadcasts
        the lookup result to every bridge in the communicator.

        Parameters
        ----------
        key : str
            The key that is being retrieved from the Analytics.
        timestep : Optional[int], optional
            Timestep associated with the requested feedback value. When
            omitted, returns the entire retained queue for ``key``.
        default : Any, optional
            Value returned when no feedback exists for ``key`` and
            ``timestep``. Defaults to ``None``.

        Notes
        -----
        This remains a collective operation when a communicator is used: all
        bridges must call ``get`` in the same order so the broadcast completes.
        The retained feedback queue is fixed-size, so old entries may be
        dropped if analytics publishes more values than the queue can hold.
        Callback execution is intentionally one timestep behind: analytics
        processes a timestep only after a later timestep or the close sentinel
        arrives. As a result, feedback for the final simulated timestep may
        only be published after ``close`` and is not meant to drive another
        simulation step.

        Returns
        -------
        Any | None
            The feedback value for ``timestep``, the full retained queue when
            ``timestep`` is omitted, or ``default`` when no feedback exists.

        Warning
        -------
        Feedback timing is asynchronous and not reproducible run to run. The
        head queue may be populated at slightly different times, and this
        bridge may read it at slightly different times. Simulation code should
        decide how to react whenever a signal becomes available, and must not
        rely on exactly when an analytics event becomes visible for simulation
        correctness.

        """
        message = None
        if self.bridge_id == 0:
            found, value = ray.get(self._get_head_actor().get_feedback.remote(key, timestep))
            message = {"found": found, "value": value}

        message = self.comm.bcast(message, root=0)
        if not message["found"]:
            return default
        return message["value"]
