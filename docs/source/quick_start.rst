Quick Start
===========

Deisa-ray (Dask Enabled In Situ Analytics with a Ray backend) lets HPC
simulations stream data into Python analytics while keeping computation close to
where the data is produced. Dask builds task graphs from your analysis code, and
Ray runs those tasks across the cluster. The result is fast, asynchronous
analytics with minimal network transfer.

.. warning::

   Do not install or use the ``distributed`` library (``dask.distributed``)
   alongside deisa-ray. DEISA installs its own Dask-on-Ray scheduler, and the
   two are not compatible. Install plain ``dask`` (as pulled in by
   ``deisa-ray``), not ``distributed``.

Assumptions and model
---------------------

Simulation side
^^^^^^^^^^^^^^^

- The simulation is distributed (MPI or similar) and iterative.
- Any rank that will **ever** send data must instantiate a ``Bridge``.
- An MPI-style communicator for all ranks is available and passed to each ``Bridge``.
- There is always a bridge with rank ``0`` (the master bridge).
- Each bridge describes all the arrays it will share at initialization via a dictionary ``arrays_metadata``.
- Sends are ordered by non-decreasing timestep: all sends for timestep *i*
  happen before any send for timestep *j > i*.
- If data is produced on GPU, copy it to CPU before calling ``Bridge.send``.

Analytics side
^^^^^^^^^^^^^^

- You instantiate a ``Deisa`` object which handles the coupling with the simulation bridges.
- You define analytics callbacks that operate on arrays sent by the simulation.
- Callback arguments are lists of ``DeisaArray`` objects (a Dask Array with a ``.t`` attribute). The MAX length of the list is the window "size" of arrays needed to perform the analytics. 
- The array name in ``Window`` must match the bridge metadata name, otherwise the callback will not run for that array.

How to run
----------

The general flow is: 

1. Install deisa-ray on all nodes (``pip install deisa-ray``). If needed,
   install MPI and mpi4py on all nodes.
2. Start a Ray cluster (head node + simulation nodes).
3. Run the analytics script on the head node.
4. Run the distributed simulation on the simulation nodes using ``mpirun`` or
   similar.

Below are example commands for each assuming that the distributed simulation is managed by MPI.

Cluster setup (Ray) 
^^^^^^^^^^^^^^^^^^^

The first step is to start the ray cluster manually: start a Ray head node on the analytics host, then join the simulation nodes.
For example (often launched via Slurm):

.. code-block:: bash

    ray start --head --address <head-node-address> # on the head node (just once)
    mpirun -hostfile <hostfile> -n <num-simulation-nodes> bash -c "ray start --address <head-node-address>" # one call per each simulation node


Analytics setup (Python)
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    mpirun -n 1 python analytics.py


Simulation setup (Python)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    mpirun -hostfile <hostfile> -n <num-simulation-nodes> python simulation.py

Example snippets
----------------

Simulation quick snippet
^^^^^^^^^^^^^^^^^^^^^^^^

The simulation creates one ``Bridge`` per participating rank and sends chunks. 

.. code-block:: python

    from mpi4py import MPI
    import numpy as np
    from deisa.ray import Bridge

    # instantiate the communicator
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()


    # descriptio of arrays being shared
    arrays_md = {
        # name of the array - must match the name used in the analytics callback!
        "temperature": {
            # shape of the full distributed array
            "global_shape": (64, 64*size),
            # shape of the chunk
            "chunk_shape": (64, 64),
            # the coordinates of the chunk block in the global distributed array
            "chunk_position": (0, rank),
        }
    }

    # Initialization of the bridge
    bridge = Bridge(
        arrays_metadata=arrays_md,
        comm=comm,
    )

    # sending chunk per timestep
    for t in range(10):
        chunk = np.ones((64, 64), dtype=np.float64) * t * rank
        bridge.send(array_name="temperature", chunk=chunk, timestep=t)

Analytics quick snippet
^^^^^^^^^^^^^^^^^^^^^^^

Define the analytics callback using Dask operations. ``DeisaArray`` provides
standard Dask array methods directly, and ``.t`` is the timestep for which that array was produced.

.. code-block:: python

    from deisa.ray import Deisa
    from deisa.ray.types import DeisaArray, Window

    d = Deisa()

    # register a callback with deisa
    @d.register(Window("temperature"))
    def summary_callback(temperature: list[DeisaArray]):
        latest = temperature[0]
        mean_value = latest.mean().compute()
        print(f"t={latest.t} mean={mean_value}")

    # register all callbacks ...

    # execute callbacks
    d.execute_callbacks()


Getting started example
-----------------------

A complete, runnable example lives in ``example/getting-started``. It starts a
small single-node Ray cluster, runs a 4-rank MPI simulation that streams a
``64 x 64`` temperature field over 5 timesteps, and registers an analytics
callback that prints the field sum and writes one PNG per timestep.

.. code-block:: bash

    cd example/getting-started
    ./launch.sh

The example needs ``mpi4py`` and ``matplotlib`` and must run with exactly 4 MPI
ranks for the simulation. See ``example/getting-started/README.md`` for what
each file does, the manual step-by-step commands, and how to change the
decomposition.

Where to go next
----------------

- ``example/getting-started/README.md`` walks through the minimal end-to-end
  example.
- :doc:`analytics` shows more callback patterns and window usage.
- The API reference under ``deisa.ray`` documents the full interface.
