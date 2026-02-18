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
- Callback arguments are lists of ``DeisaArray`` objects, one list per
  ``WindowSpec``.
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
The example metadata structure mirrors ``tests/utils.py``.

.. code-block:: python

    import numpy as np
    from deisa.ray.bridge import Bridge

    world_size = 4
    sys_md = {
        "world_size": world_size,
        "master_address": "127.0.0.1",
        "master_port": 29500,
    }

    arrays_md = {
        "temperature": {
            "chunk_shape": (64, 64),
            "nb_chunks_per_dim": (4, 4),
            "nb_chunks_of_node": 1,
            "dtype": np.float64,
            "chunk_position": (0, 0),
        }
    }

    bridge = Bridge(
        bridge_id=0,
        arrays_metadata=arrays_md,
        system_metadata=sys_md,
    )

    for t in range(10):
        chunk = np.ones((64, 64), dtype=np.float64) * t
        bridge.send(array_name="temperature", chunk=chunk, timestep=t)

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

Where to go next
----------------

- :doc:`analytics` shows more callback patterns and window usage.
- The API reference under ``deisa.ray`` documents the full interface.
