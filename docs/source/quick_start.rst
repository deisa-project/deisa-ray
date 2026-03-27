Quick Start
===========

Deisa-ray (Dask Enabled In Situ Analytics with a Ray backend) lets HPC
simulations stream data into Python analytics while keeping computation close to
where the data is produced. Dask builds task graphs from your analysis code, and
Ray runs those tasks across the cluster. The result is fast, asynchronous
analytics with minimal network transfer.

Assumptions and model
---------------------

Simulation side
^^^^^^^^^^^^^^^

- The simulation is distributed (MPI or similar) and iterative.
- Any rank that will **ever** send data must instantiate a ``Bridge``.
- The total number of participating ranks (``world_size``) is known up front.
- Each ``Bridge`` has a unique ``bridge_id``, and there is always a bridge with
  ``bridge_id=0`` (the master bridge).
- Each bridge describes the arrays it will share via ``arrays_metadata``.
- Sends are ordered by non-decreasing timestep: all sends for timestep *i*
  happen before any send for timestep *j > i*.
- If data is produced on GPU, copy it to CPU before calling ``Bridge.send``.

Analytics side
^^^^^^^^^^^^^^

- Analytics run on a Ray head node. Dask arrays are backed by Ray tasks.
- You register callbacks with ``Deisa`` and then execute them.
- Callback arguments are lists of ``DeisaArray`` objects. The length of the list is the "window size", defined when creating a ``WindowSpec``.
- The array name in ``WindowSpec`` must match the bridge metadata name,
  otherwise the callback will not run for that array.
- The window list is time-ordered and only reflects the timesteps that were
  actually sent by the simulation.

Cluster setup (Ray)
-------------------

Start a Ray head node on the analytics host, then join the simulation nodes.
For example (often launched via Slurm):

.. code-block:: bash

    ray start --head
    ray start --address <head-node-address>

Simulation quick snippet
------------------------

The simulation creates one ``Bridge`` per participating rank and sends chunks. 

.. code-block:: python

    import numpy as np
    from deisa.ray.bridge import Bridge

    # 4 Bridges in total
    world_size = 4
    sys_md = {
        "world_size": world_size,
        # must be a reachable address by all other bridges (localhost only single machine)
        "master_address": "127.0.0.1",
        # free port
        "master_port": 29500,
    }

    # descriptio of arrays being shared
    arrays_md = {
        "temperature": {
            # shape of the chunk
            "chunk_shape": (64, 64),
            # how many chunks in each dimension
            "nb_chunks_per_dim": (4, 4),
            # how many chunks / bridges per node
            "nb_chunks_of_node": 4,
            # dype
            "dtype": np.float64,
            # the coordinates of the chunk block in 
            # the global distributed array
            "chunk_position": (0, 0),
        }
    }

    # this call should be repeated 4 times with a different id
    bridge = Bridge(
        bridge_id=rank,
        arrays_metadata=arrays_md,
        system_metadata=sys_md,
    )

    # sending data chunk
    for t in range(10):
        chunk = np.ones((64, 64), dtype=np.float64) * t
        bridge.send(array_name="temperature", chunk=chunk, timestep=t)

    # close bridge
    bridge.close(timestep=10)

Analytics quick snippet
-----------------------

Define the analytics callback using Dask operations. ``DeisaArray.dask`` gives
access to standard Dask array methods, and ``DeisaArray.t`` is the timestep.

.. code-block:: python

    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import WindowSpec

    deisa = Deisa()

    def summary_callback(temperature_window):
        latest = temperature_window[-1]
        mean_value = latest.dask.mean().compute()
        print(f"t={latest.t} mean={mean_value}")

    deisa.register_callback(
        summary_callback,
        [WindowSpec("temperature", window_size=3)],
    )

    deisa.execute_callbacks()

The ``when`` keyword controls when a callback is allowed to run. By default it
is ``"AND"``, which means the callback is executed only when all required
arrays are available for the same timestep. You can also use ``when="OR"``,
which means the callback is triggered whenever any input array has new data for
a timestep; in that mode the analytics may reuse older arrays for the other
inputs.

Template:

.. code-block:: python

    deisa.register_callback(
        my_callback,
        [WindowSpec("temperature"), WindowSpec("pressure")],
        when="AND",  # or "OR"
    )

You can also use the decorator form for a shorter registration pattern:

.. code-block:: python

    d = Deisa()

    @d.callback(WindowSpec("temperature"), WindowSpec("pressure"), when="OR")
    def callback(temperature: list[DeisaArray], pressure: list[DeisaArray]):
        ...

Using ``WindowSpec`` with a sliding window
------------------------------------------

To keep the last three timesteps of an array available inside a callback, use a
``WindowSpec`` with ``window_size=3``:

.. code-block:: python

    from deisa.ray.types import WindowSpec

    temperature_spec = WindowSpec("temperature", window_size=3)

The callback argument for that window spec will contain up to the three most
recent arrays sent by the simulation. During the first two iterations, the list
will contain fewer than three arrays, so the callback should guard against
assuming the full window is already available.

The window size should be chosen based on both memory capacity and the needs of
the analysis. A window of length 3 means the system must be able to keep three
copies of that array in memory at the same time. It should also match the
algorithm you want to implement. For example, a midpoint Euler-style formula
that needs three timesteps requires ``window_size=3``.

The list is ordered from oldest to newest: the oldest array is at the
beginning, and the most recent array is at the end. Each entry is a
``DeisaArray`` object, so use ``.dask`` to access the Dask array and ``.t`` to
access its timestep.

.. code-block:: python

    def midpoint_callback(temperature_window):
        if len(temperature_window) < 3:
            return

        oldest = temperature_window[0]
        middle = temperature_window[1]
        newest = temperature_window[-1]

        midpoint_estimate = (
            oldest.dask + middle.dask + newest.dask
        ) / 3

        print(
            f"window covers timesteps {oldest.t}, {middle.t}, {newest.t}"
        )
        midpoint_estimate.compute()

Where to go next
----------------

- :doc:`analytics` shows more callback patterns and window usage.
- The API reference under ``deisa.ray`` documents the full interface.
