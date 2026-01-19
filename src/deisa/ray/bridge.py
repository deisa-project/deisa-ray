"""Bridge between MPI ranks and the Ray-based analytics system.

This module exposes the :class:`Bridge` class used by simulation ranks to
register their data chunks and exchange information with analytics running on
top of Ray.
"""

import logging
from typing import Any, Callable, Dict, Mapping, Type

import numpy as np
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from deisa.ray.scheduling_actor import SchedulingActor as _RealSchedulingActor
from deisa.ray.types import RayActorHandle

logger = logging.getLogger(__name__)


def get_node_actor_options(name: str, namespace: str) -> Dict[str, Any]:
    """Return Ray options used to create (or get) a node scheduling actor.

    Parameters
    ----------
    name : str
        Actor name to use for the node actor.
    namespace : str
        Ray namespace where the actor will live.

    Returns
    -------
    dict
        Dictionary of options to be passed to ``SchedulingActor.options``.

    Notes
    -----
    The options use ``get_if_exists=True`` to avoid race conditions when
    several bridges on the same node attempt to create the same actor.
    The actor is configured with:

    - ``lifetime='detached'`` so it survives the creating task
    - ``num_cpus=0`` so it does not reserve CPU resources
    - a very large ``max_concurrency`` because the actor is async-only
      and used mainly as a coordination point.
    """
    return {
        "name": name,
        "namespace": namespace,
        "lifetime": "detached",
        "get_if_exists": True,
        # WARNING: if not using async actor this will make OS try to spawn many threads
        # and blow everything up. Scheduling actors need to be async because of this.
        "max_concurrency": 1_000_000_000,
        "num_cpus": 0,
        "enable_task_events": False,
    }


class Bridge:
    """
    Bridge between MPI ranks and Ray cluster for distributed array processing.

    Each Bridge instance is created by an MPI rank to connect to the Ray cluster
    and send data chunks. Each Bridge is responsible for managing a chunk of data
    from the decomposed distributed array.

    Parameters
    ----------
    _node_id : str or None, optional
        The ID of the node. If None, the ID is taken from the Ray runtime context.
        Useful for testing with several scheduling actors on a single machine.
        Default is None.
    scheduling_actor_cls : Type, optional
        The class to use for creating the scheduling actor. Default is
        `_RealSchedulingActor`.
    _init_retries : int, optional
        Number of retry attempts when initializing the scheduling actor.
        Default is 3.

    Attributes
    ----------
    node_id : str
        The ID of the node this Bridge is associated with.
    scheduling_actor : RayActorHandle
        The Ray actor handle for the scheduling actor.
    preprocessing_callbacks : dict[str, Callable]
        Dictionary mapping array names to their preprocessing callback functions.

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
                "chunk_shape": (10, 10),
                "nb_chunks_per_dim": (4, 4),
                "nb_chunks_of_node": 1,
                "dtype": np.float64,
                "chunk_position": (0, 0),
            }
        }
        system_metadata = {
            "nb_ranks": 4,
            "ray_address": "auto",
        }

        bridge = Bridge(
            id=0,
            arrays_metadata=arrays_metadata,
            system_metadata=system_metadata,
        )

        bridge.send(
            array_name="temperature",
            chunk=np.zeros((10, 10), dtype=np.float64),
            timestep=0,
        )
    """

    def __init__(
        self,
        id: int,
        arrays_metadata: Mapping[str, Mapping[str, Any]],
        system_metadata: Mapping[str, Any],
        *args: Any,
        _node_id: str | None = None,
        scheduling_actor_cls: Type = _RealSchedulingActor,
        _init_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Bridge to connect MPI rank to Ray cluster.

        Parameters
        ----------
        id : int
            Unique identifier of this Bridge.
        arrays_metadata : Mapping[str, Mapping[str, Any]]
            Dictionary that describes the arrays being shared by the simulation.
            Keys represent the name of the array while the values are
            dictionaries that must at least declare the metadata expected by
            :meth:`validate_arrays_meta`.
        system_metadata : Mapping[str, Any]
            System metadata such as address of Ray cluster, number of MPI
            ranks, and other general information that describes the system.
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

        Notes
        -----
        This method automatically initializes Ray if it hasn't been initialized
        yet. The scheduling actor is created with a detached lifetime and uses
        node affinity scheduling when `_node_id` is None. The first remote call
        to the scheduling actor serves as a readiness check.
        Other Parameters
        ----------------
        *args, **kwargs
            Currently ignored. Present for backward compatibility with older
            versions of the API.
        """

        self.id = id
        self._init_retries = _init_retries

        self.arrays_metadata = self._validate_arrays_meta(arrays_metadata)
        # we add a special array with a name that will signal the end of the simulation
        # note we only need the metadata so that it can pass through the entire pipeline correctly and 
        # in sequential order, so we just replicate the first metadata we have.
        self.arrays_metadata["__deisa_last_iteration_array"] = self.arrays_metadata[list(self.arrays_metadata.keys())[0]]
        self.system_metadata = self._validate_system_meta(system_metadata)

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
        self._exchange_chunks_meta_with_node_actor()

    def _exchange_chunks_meta_with_node_actor(self):
        """
        Push per-array metadata to the node actor and cache preprocessing callbacks.

        Notes
        -----
        This method registers both global chunk layout and the bridge-specific
        chunk position for every array described in ``arrays_metadata``. It
        then fetches preprocessing callbacks from the node actor and stores
        them locally for use during :meth:`send`.
        """
        # send metadata of each array chunk (both global and local chunk info) to node actor

        for array_name, meta in self.arrays_metadata.items():
            self.node_actor.register_chunk_meta.remote(
                # global info of array (same across bridges)
                array_name=array_name,
                chunk_shape=meta["chunk_shape"],
                nb_chunks_per_dim=meta["nb_chunks_per_dim"],
                nb_chunks_of_node=meta["nb_chunks_of_node"],
                dtype=meta["dtype"],
                # local info of array specific to bridge
                bridge_id=self.id,
                chunk_position=meta["chunk_position"],
            )

        # double ray.get because method returns a ref itself
        self.preprocessing_callbacks: dict[str, Callable] = ray.get(
            ray.get(
                self.node_actor.preprocessing_callbacks.remote()  # type: ignore
            )
        )
        assert isinstance(self.preprocessing_callbacks, dict)

    def send(
        self,
        *args: Any,
        array_name: str,
        chunk: np.ndarray,
        timestep: int,
        chunked: bool = True,
        store_externally: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Make a chunk of data available to the analytics.

        This method applies the preprocessing callback associated with
        ``array_name`` to the chunk, stores it in Ray's object store, and
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
        chunked : bool, optional
            Whether the chunk was produced by the internal chunking logic.
            Currently reserved for future use. Default is True.
        store_externally : bool, optional
            If True, the data is stored externally. Not implemented yet.
            Default is False.

        Notes
        -----
        The chunk is first processed through the preprocessing callback
        associated with ``array_name``. The processed chunk is then stored in
        Ray's object store with the node actor as the owner, ensuring the
        reference persists even after the simulation script terminates.
        This method blocks until the node actor has processed the chunk.

        Raises
        ------
        KeyError
            If ``array_name`` is not found in the preprocessing callbacks
            dictionary.
        """
        # ``chunked`` and additional args/kwargs are currently reserved for
        # future extensions (e.g. multi-chunk sends). For now we only support
        # sending a single chunk described by ``arrays_metadata``.
        del args, kwargs  # explicitly unused

        chunk = self.preprocessing_callbacks[array_name](chunk)

        # Setting the owner allows keeping the reference when the simulation script terminates.
        ref = ray.put(chunk, _owner=self.node_actor)

        future: ray.ObjectRef = self.node_actor.add_chunk.remote(
            bridge_id=self.id,
            array_name=array_name,
            chunk_ref=[ref],
            timestep=timestep,
            chunked=True,
            store_externally=store_externally,
        )  # type: ignore

        # Wait until the data is processed before returning to the simulation
        ray.get(future)

    def close(
        self,
        *args: Any,
        timestep: int,
        store_externally: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Close the bridge by creating a special array that has a special name which signals to the analytics
        that the simulation has finished and that it should stop.
        """
        del args, kwargs  # explicitly unused
        ref = ray.put(0, _owner=self.node_actor)
        future: ray.ObjectRef = self.node_actor.add_chunk.remote(
            bridge_id=self.id,
            array_name="__deisa_last_iteration_array",
            chunk_ref=[ref],
            timestep=timestep,
            chunked=True,
            store_externally=store_externally,
        )  # type: ignore
        ray.get(future)

    def _validate_arrays_meta(
        self,
        arrays_metadata: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """
        Validate and normalize the ``arrays_metadata`` argument.

        Parameters
        ----------
        arrays_metadata : Mapping[str, Mapping[str, Any]]
            User-provided metadata for all arrays handled by this bridge.

        Returns
        -------
        dict[str, dict[str, Any]]
            A shallow-copied and validated version of the input mapping.

        Raises
        ------
        TypeError
            If the top-level mapping, keys or values have incorrect types.
        ValueError
            If required keys are missing for any array.
        """
        if not isinstance(arrays_metadata, Mapping):
            raise TypeError(f"arrays_metadata must be a mapping from str to dict, got {type(arrays_metadata).__name__}")

        required_keys = {
            "chunk_shape",
            "nb_chunks_per_dim",
            "nb_chunks_of_node",
            "dtype",
            "chunk_position",
        }

        validated: dict[str, dict[str, Any]] = {}

        for array_name, meta in arrays_metadata.items():
            # key type
            if not isinstance(array_name, str):
                raise TypeError(f"arrays_metadata keys must be str, got {type(array_name).__name__}")

            # value type
            if not isinstance(meta, Mapping):
                raise TypeError(f"arrays_metadata['{array_name}'] must be a mapping, got {type(meta).__name__}")

            # required keys present?
            missing = required_keys - meta.keys()
            if missing:
                raise ValueError(f"arrays_metadata['{array_name}'] is missing required keys: {missing}")

            self._validate_single_array_metadata(array_name, meta)
            validated[array_name] = dict(meta)

        return validated

    def _validate_single_array_metadata(
        self,
        name: str,
        meta: Mapping[str, Any],
    ) -> None:
        """
        Validate metadata for a single array entry.

        Parameters
        ----------
        name : str
            Array name.
        meta : Mapping[str, Any]
            Metadata for this array. Must contain at least:

            - ``chunk_shape``: sequence of positive ints
            - ``nb_chunks_per_dim``: sequence of positive ints
            - ``nb_chunks_of_node``: positive int
            - ``dtype``: NumPy dtype or anything accepted by ``np.dtype``
            - ``chunk_position``: sequence of ints of same length as
              ``chunk_shape``

        Raises
        ------
        TypeError
            If any field has an invalid type.
        ValueError
            If shapes/positions have inconsistent lengths.
        """
        # chunk_shape: tuple/list of positive ints
        chunk_shape = meta["chunk_shape"]
        if not (isinstance(chunk_shape, (tuple, list)) and all(isinstance(n, int) and n > 0 for n in chunk_shape)):
            raise TypeError(
                f"arrays_metadata['{name}']['chunk_shape'] must be a sequence of positive ints, got {chunk_shape!r}"
            )

        # nb_chunks_per_dim: same pattern
        nb_chunks_per_dim = meta["nb_chunks_per_dim"]
        if not (
            isinstance(nb_chunks_per_dim, (tuple, list))
            and all(isinstance(n, int) and n > 0 for n in nb_chunks_per_dim)
        ):
            raise TypeError(
                f"arrays_metadata['{name}']['nb_chunks_per_dim'] must be a "
                f"sequence of positive ints, got {nb_chunks_per_dim!r}"
            )

        # nb_chunks_of_node: positive int
        nb_chunks_of_node = meta["nb_chunks_of_node"]
        if not (isinstance(nb_chunks_of_node, int) and nb_chunks_of_node > 0):
            raise TypeError(
                f"arrays_metadata['{name}']['nb_chunks_of_node'] must be a positive int, "
                f"got {type(meta['nb_chunks_of_node']).__name__}"
            )

        # chunk_position: sequence of ints of same length as chunk_shape (optional)
        chunk_position = meta["chunk_position"]
        if not (
            isinstance(chunk_position, (tuple, list))
            and all(
                isinstance(pos, int) and 0 <= pos < nb_chunks
                for pos, nb_chunks in zip(chunk_position, nb_chunks_per_dim)
            )
        ):
            raise TypeError(
                f"arrays_metadata['{name}']['chunk_position'] must be a sequence of ints, got {chunk_position!r}"
            )

        if len(chunk_position) != len(meta["chunk_shape"]):
            raise ValueError(f"arrays_metadata['{name}']['chunk_position'] must have the same length as 'chunk_shape'")

    def _validate_system_meta(self, system_meta: Mapping[str, Any]) -> dict[str, Any]:
        """
        Validate and normalize the ``system_metadata`` argument.

        Parameters
        ----------
        system_meta : Mapping[str, Any]
            User-provided system-level metadata.

        Returns
        -------
        dict[str, Any]
            A shallow-copied version of the input mapping.

        Raises
        ------
        TypeError
            If ``system_meta`` is not a mapping.
        """
        if not isinstance(system_meta, Mapping):
            raise TypeError(f"system_metadata must be a mapping, got {type(system_meta).__name__}")
        return dict(system_meta)

    def _create_node_actor(
        self,
        node_actor_cls: Type = _RealSchedulingActor,
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

    # TODO feedback needs testing
    def get(
        self,
        *args: Any,
        name: str,
        default: Any | None = None,
        chunked: bool = False,
        **kwargs: Any,
    ) -> Any | None:
        """
        Retrieve information back from Analytics.

        Used for two cases:

        1. Retrieve a simple value that is set in the analytics so that the
           simulation can react to some event that has been detected. This
           case is asynchronous.
        2. (Planned) Retrieve a distributed array that has been modified by
           the analytics. This case is synchronous and currently not
           implemented for ``chunked=True``.

        Parameters
        ----------
        name : str
            The name of the key that is being retrieved from the Analytics.
        default : Any, optional
            The default value to return if the key has not been set or does
            not exist. Default is None.
        chunked : bool, optional
            Whether the value that is returned is distributed or not. Should
            be set to True only if retrieving a distributed array that is
            handled by the bridge. Currently not implemented. Default is
            False.

        Notes
        -----
        When ``chunked`` is False, this method simply forwards the request to
        the node actor and returns the result (or ``default`` if not set).
        When ``chunked`` is True, a :class:`NotImplementedError` is raised.
        """
        del args, kwargs  # explicitly unused

        if not chunked:
            return ray.get(self.node_actor.get.remote(name, default, chunked))

        raise NotImplementedError("Retrieving chunked arrays via Bridge.get is not implemented yet.")

    def _delete(self, *args: Any, name: str, **kwargs: Any) -> None:
        """
        Delete a feedback key from the node actor if present.

        Parameters
        ----------
        name : str
            Key to remove.

        Notes
        -----
        This is currently used internally after :meth:`get` to avoid
        repeatedly signaling the same event. Missing keys are silently
        ignored.
        """
        del args, kwargs  # explicitly unused
        # Currently the semantics of deletion are still under discussion. For
        # now, delegate to the node actor which maintains the shared state.
        self.node_actor.delete.remote(name)
