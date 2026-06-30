Analytics
=========

In callback examples, the name passed to ``Window`` is also the keyword
argument name used when DEISA calls the callback. Each argument is a
``list[DeisaArray]`` ordered from oldest to newest; with no explicit
``size`` the list contains only the latest shared timestep.

Simple example
--------------

.. code-block:: python

    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import DeisaArray, Window

    d = Deisa()

    @d.register(Window("temperature"))
    def summarize_temperature(temperature: list[DeisaArray]):
        mean_temperature = temperature[0].mean().compute()
        print("Mean temperature:", mean_temperature)

    d.execute_callbacks()

Several arrays
--------------

.. code-block:: python

    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import DeisaArray, Window

    d = Deisa()

    @d.register(Window("temperature"), Window("pressure"))
    def compare_state(
        temperature: list[DeisaArray],
        pressure: list[DeisaArray],
    ):
        thermal_pressure_balance = (
            temperature[0] - pressure[0]
        ).mean().compute()
        print("thermal-pressure balance:", thermal_pressure_balance)

    d.execute_callbacks()

``when="AND"`` and ``when="OR"``
--------------------------------

When a callback depends on several arrays, ``when`` controls which arrivals
are allowed to trigger it.

Before applying ``when``, DEISA first groups shares by timestep. It keeps
collecting arrays for the current timestep and does not analyze callbacks for
that timestep until it receives any share from a higher timestep, or the final
close sentinel. The higher-timestep share acts as the signal that the previous
timestep is complete enough to inspect.

.. code-block:: text

    incoming shares:

    event      share received                  DEISA action
    1          temperature, t = 1              collect t = 1; do not call yet
    2          pressure, t = 1                 collect t = 1; do not call yet
    3          velocity, t = 1                 collect t = 1; do not call yet
    4          temperature, t = 2              now analyze callbacks for t = 1
                                                keep temperature, t = 2 for later

After that boundary is reached, DEISA uses the callback's ``when`` value to
decide whether the callback should run for the timestep being analyzed.
``when`` is evaluated per callback, using only the arrays requested by that
callback.

``when="AND"`` is the default. The callback runs only if all arrays requested
by that callback have a new share for the timestep being analyzed. If a
callback asks for temperature, pressure, and velocity, but only temperature and
pressure were shared for timestep 2, the callback is not called.

``when="OR"`` runs if at least one array requested by the callback has a new
share for the timestep being analyzed. Arrays that did not receive a new share
reuse their most recent available window, after they have been seen at least
once. If none of the arrays requested by a callback have a new share, the
callback is not called in either ``AND`` or ``OR`` mode.

.. code-block:: text

    Callback inputs:
        Window("temperature"), Window("pressure"), Window("velocity")

    timestep being analyzed      new shares seen          when="AND"        when="OR"
    t = 1                        temperature, pressure,   run               run
                                 velocity
    t = 2                        temperature, pressure    do not run        run
    t = 3                        none of these arrays     do not run        do not run

In both modes, callback arguments are still lists of ``DeisaArray`` objects.
With ``when="OR"``, a list may be reused from an older timestep if that array
did not produce a new share for the current trigger.

Sliding window
--------------

If the analysis requires access to several iterations (for example, to compute
a time derivative), it is possible to use the ``size`` parameter.
Choose the window size from both the available system memory and the analytics
algorithm you want to run. A ``size`` of 5 means DEISA may need to keep
five timesteps of that array available, so it costs more memory than a smaller
window. It is the right choice for an analysis that needs five temporal points,
such as a five-point stencil finite-difference approximation.

The window size is an upper bound on what the callback receives, not a
requirement that every operation in the callback must consume the whole window.
A callback registered with ``size=5`` can still compute single-timestep
statistics from ``temperature[-1]``, three-timestep estimates from the newest
three entries, and five-timestep estimates only when all five entries are
available.

Callbacks are called as soon as their input arrays are available, even before
the sliding window is full. During the first few calls, the list may contain
fewer entries than ``size``. The callback must check ``len(window)`` for
any operation that needs a minimum number of timesteps.

.. code-block:: python

    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import DeisaArray, Window

    d = Deisa()

    @d.register(Window("temperature", size=5))
    def estimate_temperature_change(temperature: list[DeisaArray]):
        latest_mean = temperature[-1].mean().compute()
        print("mean temperature:", latest_mean)

        if len(temperature) >= 3:
            newest = temperature[-1]
            middle = temperature[-2]
            oldest = temperature[-3]
            three_point_rate = (
                newest - oldest
            ) / (newest.t - oldest.t)
            print("three-point mean dT/dt:", three_point_rate.mean().compute())

        if len(temperature) < 5:
            return

        five_point_average = sum(
            timestep for timestep in temperature
        ) / 5
        print("five-point average:", five_point_average.mean().compute())

    d.execute_callbacks()

Dask persist
------------

Dask's ``persist`` is supported:

.. code-block:: python

    import dask.array as da
    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import DeisaArray, Window

    d = Deisa()

    @d.register(Window("vorticity"))
    def track_vorticity(vorticity: list[DeisaArray]):
        total_vorticity = vorticity[0].sum().persist()

        # The result is still a Dask array, but the sum is computing in the background.
        assert isinstance(total_vorticity, da.Array)

        print("t=", vorticity[0].t, "total vorticity=", total_vorticity.compute())

    d.execute_callbacks()

Saving to HDF5
--------------

``DeisaArray`` provides a convenience method for writing one array to HDF5:

.. code-block:: python

    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import DeisaArray, Window

    d = Deisa()

    @d.register(Window("temperature"))
    def save_hotspot_temperature(temperature: list[DeisaArray]):
        if temperature[0].t == 5:
            temperature[0].to_hdf5("interesting-event.h5", "temperature")

    d.execute_callbacks()

If you want to save several arrays into the same HDF5 file, use
``deisa.ray.types.to_hdf5``:

.. code-block:: python

    from deisa.ray.types import DeisaArray, Window, to_hdf5
    from deisa.ray.window_handler import Deisa

    d = Deisa()

    @d.register(Window("temperature"), Window("pressure"))
    def save_state_snapshot(
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
    from deisa.ray.types import DeisaArray, Window

    d = Deisa()

    @d.register(Window("temperature"))
    def inspect_temperature_field(temperature: list[DeisaArray]):
        temperature_da = xr.DataArray(
            temperature[0],
            dims=["x", "y"],
            name="temperature",
        )

        print("timestep:", temperature[0].t)
        print(temperature_da)

    d.execute_callbacks()

Saving Xarray to NetCDF
-----------------------

One convenient pattern is to convert the ``DeisaArray`` to an
``xarray.DataArray`` and then write it to NetCDF:

.. code-block:: python

    import xarray as xr
    from deisa.ray.window_handler import Deisa
    from deisa.ray.types import DeisaArray, Window

    d = Deisa()

    @d.register(Window("temperature"))
    def save_temperature_netcdf(temperature: list[DeisaArray]):
        if temperature[0].t == 5:
            xarray_da = xr.DataArray(
                temperature[0],
                dims=["x", "y"],
                name="temperature",
            ).compute()

            xarray_da.to_netcdf("interesting-event.nc")

    d.execute_callbacks()
