Analytics
=========

Simple example
--------------

.. code-block:: python

    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import WindowSpec

    deisa = Deisa()

    def simulation_callback(array: da.Array, timestep: int):
        x = array.sum().compute()
        print("Sum:", x)

    deisa.register_callback(
        simulation_callback,
        [WindowSpec("array")],
    )
    deisa.execute_callbacks()

Several arrays
--------------

.. code-block:: python

    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import WindowSpec

    deisa = Deisa()

    def simulation_callback(a: da.Array, b: da.Array, timestep: int):
        r = (a - b).mean().compute()

    deisa.register_callback(
        simulation_callback,
        [WindowSpec("a"), WindowSpec("b")]
    )
    deisa.execute_callbacks()

Sliding window
--------------

If the analysis requires access to several iterations (for example, to compute
a time derivative), it is possible to use the ``window_size`` parameter.

.. code-block:: python

    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import WindowSpec

    deisa = Deisa()

    def simulation_callback(arrays: list[da.Array], timestep: int):
        if len(arrays) < 2:  # For the first iteration
            return

        current_array = arrays[1]
        previous_array = arrays[0]

        ...

    deisa.register_callback(
        simulation_callback,
        [
            WindowSpec("array", window_size=2),  # Enable sliding window
        ],
    )
    deisa.execute_callbacks()

Dask persist
------------

Dask's ``persist`` is supported:

.. code-block:: python

    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import WindowSpec

    deisa = Deisa()

    def simulation_callback(array: da.Array, timestep: int):
        x = array.sum().persist()

        # x is still a Dask array, but the sum is being computed in the background
        assert isinstance(x, da.Array)

        x_final = x.compute()
        assert x_final == 10 * timestep

    deisa.register_callback(
        simulation_callback,
        [WindowSpec("array")],
        max_iterations=NB_ITERATIONS,
    )
    deisa.execute_callbacks()

