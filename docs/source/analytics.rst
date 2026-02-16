Analytics
=========

Simple example
--------------

.. code-block:: python

    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import WindowSpec

    deisa = Deisa()

    def simulation_callback(array: list[DeisaArray]):
        x = array[0].dask.sum().compute()
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

    def simulation_callback(a: list[DeisaArray], b: list[DeisaArray]):
        r = (a[0].dask - a[0].dask).sum().compute()

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

    def simulation_callback(arrays: list[DeisaArray]):
        if len(arrays) < 2:  # For the first iteration
            return

        current_array = arrays[1].dask
        previous_array = arrays[0].dask

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

    def simulation_callback(array: list[DeisaArray]):
        x = array[0].dask.sum().persist()

        # x is still a Dask array, but the sum is being computed in the background
        assert isinstance(x, da.Array)

        x_final = x.compute()
        assert x_final == 10 * timestep

    deisa.register_callback(
        simulation_callback,
        [WindowSpec("array")],
    )

    deisa.execute_callbacks()

