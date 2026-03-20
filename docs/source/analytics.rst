Analytics
=========

Simple example
--------------

.. code-block:: python

    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import DeisaArray, WindowSpec

    d = Deisa()

    @d.callback(WindowSpec("array"))
    def simulation_callback(array: list[DeisaArray]):
        x = array[0].dask.sum().compute()
        print("Sum:", x)

    d.execute_callbacks()

Several arrays
--------------

.. code-block:: python

    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import DeisaArray, WindowSpec

    d = Deisa()

    @d.callback(WindowSpec("a"), WindowSpec("b"))
    def simulation_callback(a: list[DeisaArray], b: list[DeisaArray]):
        r = (a[0].dask - b[0].dask).sum().compute()

    d.execute_callbacks()

Sliding window
--------------

If the analysis requires access to several iterations (for example, to compute
a time derivative), it is possible to use the ``window_size`` parameter.

.. code-block:: python

    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import DeisaArray, WindowSpec

    d = Deisa()

    @d.callback(WindowSpec("array", window_size=2))
    def simulation_callback(arrays: list[DeisaArray]):
        if len(arrays) < 2:  # For the first iteration
            return

        current_array = arrays[1].dask
        previous_array = arrays[0].dask

        ...

    d.execute_callbacks()

Dask persist
------------

Dask's ``persist`` is supported:

.. code-block:: python

    import dask.array as da
    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import DeisaArray, WindowSpec

    d = Deisa()

    @d.callback(WindowSpec("array"))
    def simulation_callback(array: list[DeisaArray]):
        x = array[0].dask.sum().persist()

        # x is still a Dask array, but the sum is being computed in the background
        assert isinstance(x, da.Array)

        print("t=", array[0].t, "sum=", x.compute())

    d.execute_callbacks()

Saving to HDF5
--------------

``DeisaArray`` provides a convenience method for writing one array to HDF5:

.. code-block:: python

    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import DeisaArray, WindowSpec

    d = Deisa()

    @d.callback(WindowSpec("temperature"))
    def save_temperature(array: list[DeisaArray]):
        if array[0].t == 5:
            array[0].to_hdf5("interesting-event.h5", "temperature")

    d.execute_callbacks()

If you want to save several arrays into the same HDF5 file, use
``deisa.ray.types.to_hdf5``:

.. code-block:: python

    from deisa.ray.types import DeisaArray, WindowSpec, to_hdf5
    from deisa.ray.window_handler import Deisa

    d = Deisa()

    @d.callback(WindowSpec("temperature"), WindowSpec("pressure"))
    def save_both(
        temperature: list[DeisaArray],
        pressure: list[DeisaArray],
    ):
        if temperature[0].t == 5:
            to_hdf5(
                "state.h5",
                {
                    "temperature": temperature[0],
                    "pressure": pressure[0],
                },
            )

    d.execute_callbacks()

Converting to Xarray
--------------------

If you want to work with Xarray APIs, build an ``xarray.DataArray`` from the
underlying Dask array:

.. code-block:: python

    import xarray as xr
    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import DeisaArray, WindowSpec

    d = Deisa()

    @d.callback(WindowSpec("temperature"))
    def xarray_example(array: list[DeisaArray]):
        da = xr.DataArray(
            array[0].dask,
            dims=["x", "y"],
            name="temperature",
        )

        print("timestep:", array[0].t)
        print(da)

    d.execute_callbacks()

Saving Xarray to NetCDF
-----------------------

One convenient pattern is to convert the ``DeisaArray`` to an
``xarray.DataArray`` and then write it to NetCDF:

.. code-block:: python

    import xarray as xr
    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import DeisaArray, WindowSpec

    d = Deisa()

    @d.callback(WindowSpec("temperature"))
    def save_netcdf(array: list[DeisaArray]):
        if array[0].t == 5:
            xarray_da = xr.DataArray(
                array[0].dask,
                dims=["x", "y"],
                name="temperature",
            ).compute()

            xarray_da.to_netcdf("interesting-event.nc")

    d.execute_callbacks()
