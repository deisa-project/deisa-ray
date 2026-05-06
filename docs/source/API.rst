User API
========

This page describes the user-facing API of ``deisa-ray``. It is written from
the behavior of this repository, not only from the ``deisa-core``
protocols. The core model is intentionally small:

- simulation processes create one ``Bridge`` per participating rank and call
  ``send`` whenever a local chunk is ready and the simulation chooses to share
  that timestep with analytics;
- analytics code creates one ``Deisa`` object, registers callbacks, and calls
  ``execute_callbacks``;
- callbacks receive lists of ``DeisaArray`` objects, where each object wraps a
  Dask array and the timestep it came from.

Think of ``send`` as the point where the simulation exposes a timestep to
DEISA. For a distributed array at a given timestep, all bridges that own chunks
of that array must share their local chunk for the same timestep before DEISA
can assemble the full Dask array seen by analytics.

Callbacks see those shares through a sliding window. A ``Window`` with
``window_size=N`` gives the callback up to the most recent ``N`` shared
timesteps for that array, ordered oldest to newest. This is what lets analysis
code combine data across time, for example comparing the newest field with the
previous one or computing a derivative over several timesteps.

The simulation and analytics may start in either order. The runtime uses Ray
actors to rendezvous, collect chunks, build Dask arrays, and execute Dask graphs
close to the data.

Bridges also use a communicator to coordinate with each other. By default,
``deisa-ray`` creates a Gloo communicator from ``system_metadata``, but
applications that already run under MPI can pass an MPI communicator instead.
That communicator is used for fast, efficient bridge-to-bridge coordination,
including barriers and feedback broadcasts.

Main imports
------------

.. code-block:: python

    from deisa.ray.bridge import Bridge
    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import DeisaArray, Window, to_hdf5

Simulation-side API
-------------------

``Bridge``
^^^^^^^^^^

``Bridge`` is the simulation-side entry point.

.. code-block:: python

    bridge = Bridge(
        bridge_id=rank,
        arrays_metadata=arrays_metadata,
        system_metadata=system_metadata,
    )

Each rank that will ever send data must create a ``Bridge``. The
``bridge_id`` must be unique among participating ranks, and there must be a
bridge with ``bridge_id=0``. Bridge ``0`` is special because it sends the final
sentinel array used to stop analytics callback execution.

Arguments
"""""""""

``bridge_id``
    Integer identifier for this simulation rank. In MPI programs this is
    usually the MPI rank.

``arrays_metadata``
    Mapping from array name to metadata for the chunk owned by this bridge.
    Each value must contain:

    ``chunk_shape``
        Shape of the local chunk as a tuple/list/1D NumPy array of positive
        integers.

    ``nb_chunks_per_dim``
        Number of chunks in each dimension of the global array. The full array
        shape is approximately ``chunk_shape * nb_chunks_per_dim`` dimension by
        dimension.

    ``nb_chunks_of_node``
        Number of chunks for this array expected on the current scheduling
        actor. This is the local completeness count used before forwarding an
        array timestep.

    ``dtype``
        NumPy dtype, or a value accepted by ``numpy.dtype``.

    ``chunk_position``
        Position of this bridge's chunk in the global chunk grid. It must have
        the same dimensionality as ``chunk_shape`` and each index must be within
        ``nb_chunks_per_dim``.

``system_metadata``
    Required when ``comm`` is omitted. The default communicator uses
    ``system_metadata["world_size"]``, ``system_metadata["master_address"]``,
    and ``system_metadata["master_port"]`` to initialize a Gloo process group.
    If a custom communicator is supplied, this argument may be omitted.

``comm``
    Optional communicator. A raw ``mpi4py.MPI.Comm`` is accepted and wrapped
    automatically. Any custom communicator must expose ``rank``, ``world_size``,
    ``barrier()``, and ``broadcast_object(obj, src=0)``. ``Bridge.get`` uses
    ``broadcast_object`` so bridge ``0`` can query feedback once and share the
    result with all participating bridges. Passing an MPI communicator is the
    recommended option when the simulation already has one, because it reuses
    the simulation's native fast communication layer instead of creating the
    default Gloo process group.

``_node_id``, ``scheduling_actor_cls``, ``_init_retries``, ``_comm_timeout``
    Implementation and testing hooks. Normal users should not need them.
    ``_comm_timeout`` controls the default Gloo rendezvous timeout in seconds.

Behavior
""""""""

During construction, ``Bridge`` validates array metadata, initializes or
normalizes its communicator, starts Ray with ``address="auto"`` if needed,
creates or reuses a detached scheduling actor on the local Ray node, registers
the chunk metadata with that actor, waits for actor readiness, and finally
enters a communicator barrier so all bridges start from a consistent point.

If the default Gloo communicator is used and not all bridge processes connect
before the timeout, construction raises the underlying torch distributed
rendezvous error with extra context about the expected rank count and master
address.

``Bridge.send``
^^^^^^^^^^^^^^^

.. code-block:: python

    bridge.send(array_name="temperature", chunk=chunk, timestep=t)

``send`` makes one local chunk available to analytics.

Arguments
"""""""""

``array_name``
    Name of the array. It must match a key in ``arrays_metadata`` and the
    analytics-side ``Window`` name.

``chunk``
    A ``numpy.ndarray`` containing this bridge's local chunk. If the simulation
    computes on GPU, copy the data to CPU before calling ``send``.

``timestep``
    Integer timestep for the chunk.

``test_mode``
    Reserved test hook and currently ignored.

Behavior
""""""""

``send`` stores the chunk in Ray's object store with the scheduling actor as
owner, forwards the object reference to the scheduling actor, and blocks until
that actor has processed the chunk. If the scheduling layer detects a contract
violation, ``send`` raises ``ContractError``.

The timestep stream must be non-decreasing globally: all data for timestep
``i`` must be sent before any data for timestep ``j > i``. The analytics loop
checks this ordering and raises if it observes an older timestep after a newer
one.

``Bridge.close``
^^^^^^^^^^^^^^^^

.. code-block:: python

    bridge.close(timestep=final_timestep)

``close`` tells analytics that the simulation has finished. All bridges must
call it because it begins with a communicator barrier. After the barrier,
bridge ``0`` sends an internal sentinel array named
``"__deisa_last_iteration_array"``. Analytics stop when that sentinel is
received.

``timestep`` is the timestep attached to the sentinel. A common pattern is to
use the first timestep after the last real send. ``close`` returns the same
integer.

``Bridge.get``
^^^^^^^^^^^^^^

.. code-block:: python

    value = bridge.get("cooling_factor", timestep=t)

``get`` retrieves a timestamped feedback value published by analytics with
``Deisa.set``. It is a collective operation: when a communicator is used, every
bridge must call ``get`` in the same order for a given key/timestep lookup.
Bridge ``0`` queries the global head actor and broadcasts the result to the
other bridges.

Arguments
"""""""""

``name``
    Feedback key to read.

``timestep``
    Optional timestep associated with the requested feedback value. When it is
    provided, ``get`` returns the value for exactly that timestep, or ``None``
    if no value is currently retained for that key/timestep. When omitted,
    ``get`` returns the retained queue for ``name`` as a list of
    ``(timestep, value)`` pairs, or ``None`` if no feedback exists for the key.

Behavior
""""""""

Feedback values are stored in fixed-size per-key queues on the head actor.
Older entries may be dropped after analytics publishes more values than the
queue can hold. A missing value returns ``None``; there is no ``default``
argument in the current API.

Feedback is asynchronous and intentionally opportunistic. Analytics callbacks
run after DEISA has observed a later timestep, or the close sentinel, so
feedback for the final simulated timestep may only become visible during
shutdown. Simulation correctness should not depend on observing an analytics
event at one exact timestep.

Analytics-side API
------------------

``Deisa``
^^^^^^^^^

``Deisa`` is the analytics-side entry point.

.. code-block:: python

    deisa = Deisa(max_simulation_ahead=1, feedback_queue_size=1024)

Arguments
"""""""""

``ray_start``
    Optional callable used to start Ray. If omitted, ``Deisa`` calls
    ``ray.init(address="auto", log_to_driver=False, logging_level=logging.ERROR)``
    when it first connects.

``max_simulation_ahead``
    Number of timesteps the simulation may be ahead of analytics before
    scheduling applies back-pressure. The default is ``1``.

``feedback_queue_size``
    Maximum number of feedback entries retained per key on the head actor.
    The default is ``1024``. It must be greater than zero.

Behavior
""""""""

The constructor is intentionally cheap: it records configuration but does not
start Ray. Ray connection, Dask scheduler configuration, and head actor creation
happen lazily when registering callbacks or executing them.

``Deisa.register_callback``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def summary(temperature: list[DeisaArray]):
        latest = temperature[-1]
        print(latest.t, latest.mean().compute())

    deisa.register_callback(
        summary,
        [Window("temperature", window_size=3)],
        when="AND",
    )

Registers a callback to be called from ``execute_callbacks``.

Arguments
"""""""""

``simulation_callback``
    Callable that receives keyword arguments named after each ``Window``.
    Each argument is a ``list[DeisaArray]``. For ergonomic callbacks, choose
    array names that are valid Python parameter names, or write the callback to
    accept ``**kwargs``.

``arrays_spec``
    List of ``Window`` objects describing which arrays the callback needs
    and how many timesteps should be kept for each array.

``exception_handler``
    Optional callable invoked when the user callback raises. If omitted, the
    default handler prints/logs the exception and callback execution continues.
    ``TimeoutError`` and ``AssertionError`` are re-raised by the execution loop.

``when``
    ``"AND"`` or ``"OR"``. With ``"AND"``, the callback runs only when every
    requested array has new data for the current timestep. With ``"OR"``, the
    callback runs after all requested arrays have been seen at least once and
    at least one requested array has new data; arrays without new data reuse
    their latest window.

Returns
"""""""

Returns the original callback so the method can be used by decorator helpers
and by code that wants to keep the callable.

``Deisa.register``
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    @deisa.register(Window("temperature"), Window("pressure"), when="OR")
    def compare(temperature: list[DeisaArray], pressure: list[DeisaArray]):
        ...

Decorator form of ``register_callback``. It accepts ``Window`` objects as
positional arguments and the same ``exception_handler`` and ``when`` keyword
arguments.

``Deisa.execute_callbacks``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    deisa.execute_callbacks()

Starts the analytics loop. At least one callback must be registered first,
otherwise ``RuntimeError`` is raised.

Behavior
""""""""

``execute_callbacks`` marks analytics as ready, builds one internal queue per
array, waits for arrays from the head actor, and calls every registered
callback whose ``when`` condition is satisfied. The loop ends when bridge ``0``
sends the internal final sentinel through ``Bridge.close``.

Callback arguments are always lists. With no explicit window size, the list has
one element: the latest ``DeisaArray``. With ``window_size=N``, the list
contains up to the last ``N`` arrays for that name, ordered oldest to newest.
Early timesteps may have shorter lists, so callbacks that require a full
window should check ``len(window)`` before computing.

``Deisa.set``
^^^^^^^^^^^^^

.. code-block:: python

    deisa.set("cooling_factor", value=0.5, timestep=latest.t)

Publishes a timestamped feedback value from analytics so simulation bridges can
retrieve it collectively with ``Bridge.get``.

Arguments
"""""""""

``key``
    Hashable feedback key.

``value``
    Python object to store.

``timestep``
    Timestep associated with ``value``. For a given key, feedback timesteps
    must be strictly increasing. Publishing the same timestep twice, or
    publishing an older timestep after a newer one, raises ``ValueError``.
    Timesteps for a key must also be mutually comparable; otherwise the order
    check raises ``TypeError``.

Behavior
""""""""

``set`` ensures the analytics side is connected, then stores ``(timestep,
value)`` on the head actor's fixed-size feedback queue for ``key``. The call
waits for the head actor to accept the value, but simulation-side observation
is still asynchronous because bridges poll/read feedback independently of
callback execution.

Callback data types
-------------------

``Window``
^^^^^^^^^^^^^^

.. code-block:: python

    Window("temperature")
    Window("temperature", window_size=3)

``Window`` describes one callback input.

``name``
    Array name. It must match ``Bridge`` metadata and ``Bridge.send``.

``window_size``
    Number of recent timesteps to pass to the callback. ``None`` means only the
    latest array is passed. A positive integer creates a sliding window with up
    to that many entries.

The runtime allocates internal queues large enough for the largest requested
window per array across all registered callbacks.

``DeisaArray``
^^^^^^^^^^^^^^

``DeisaArray`` is the object passed inside callback windows.

``dask``
    A ``dask.array.Array`` representing the full distributed array for one
    timestep. Use normal Dask operations, then call ``compute`` or ``persist``
    when you want execution.

``t``
    Integer timestep associated with the array.

``to_hdf5(fname, dataset)``
    Convenience method that writes this array to an HDF5 virtual dataset.

``to_zarr(...)``
    Convenience wrapper around ``dask.array.to_zarr``. The array is persisted
    before writing.

``to_hdf5``
^^^^^^^^^^^

.. code-block:: python

    to_hdf5(
        "state.h5",
        {
            "temperature": temperature[-1],
            "pressure": pressure[-1],
        },
    )

Writes one or more ``DeisaArray`` objects to one HDF5 file using HDF5 virtual
datasets. Chunk payloads are written to hidden chunk files named from the final
file, dataset name, and chunk position; the final HDF5 file links those chunks
through VDS. Dataset names with unusual filesystem characters may produce
awkward chunk filenames, so simple dataset names are recommended.

Execution guarantees and assumptions
------------------------------------

Array names
    The same array name must be used in ``arrays_metadata``, ``Bridge.send``,
    and ``Window``. Analytics callbacks receive keyword arguments with
    those names.

Participating ranks
    Any rank that will ever send data must instantiate a ``Bridge``. The
    participating world size is fixed at startup.

Master bridge
    ``bridge_id=0`` must exist. It is responsible for emitting the final
    sentinel in ``Bridge.close``.

Timestep ordering
    Timesteps must be sent in non-decreasing order. The system assumes it can
    process all arrays for timestep ``i`` before processing timestep ``i+1``.

Window ordering
    Callback windows are ordered oldest to newest. The most recent item is
    ``window[-1]``.

Collective operations
    Bridge construction with the default communicator and ``Bridge.close`` are
    collective over participating bridges. ``Bridge.get`` is also collective
    when a communicator is used because bridge ``0`` broadcasts the lookup
    result. All participating bridges must enter these operations in compatible
    order.

Feedback ordering
    ``Deisa.set`` stores timestamped feedback in per-key queues. For each key,
    feedback timesteps must be strictly increasing. ``Bridge.get`` returns
    ``None`` for missing values, and may miss values that are too old for the
    retained queue.

Data locality
    ``Bridge.send`` stores chunks in Ray's object store under the local
    scheduling actor. Dask graphs are then scheduled by Ray, with optional
    experimental distributed scheduling controlled by project configuration.

Current limitations
-------------------

- Feedback is for small Python objects. Distributed-array feedback from
  analytics back to simulation is not part of the current user API.
- ``Bridge.send`` expects CPU ``numpy.ndarray`` inputs.
- The theoretical ``deisa-core`` API includes protocol methods such as
  ``get_array``, ``delete``, and ``close`` on the analytics object. Those are
  not the concrete user API of this Ray implementation today.

Minimal end-to-end shape
------------------------

Simulation:

.. code-block:: python

    bridge = Bridge(
        bridge_id=rank,
        arrays_metadata={
            "temperature": {
                "chunk_shape": (64, 64),
                "nb_chunks_per_dim": (4, 4),
                "nb_chunks_of_node": 4,
                "dtype": np.float64,
                "chunk_position": chunk_position,
            },
        },
        system_metadata={
            "world_size": world_size,
            "master_address": "127.0.0.1",
            "master_port": 29500,
        },
    )

    for timestep in range(10):
        bridge.send(
            array_name="temperature",
            chunk=make_temperature_chunk(timestep),
            timestep=timestep,
        )

    bridge.close(timestep=10)

Analytics:

.. code-block:: python

    deisa = Deisa()

    @deisa.register(Window("temperature", window_size=3))
    def analyze_temperature(temperature: list[DeisaArray]):
        if len(temperature) < 3:
            return

        newest = temperature[-1]
        mean_value = newest.mean().compute()
        print("timestep", newest.t, "mean", mean_value)

    deisa.execute_callbacks()
