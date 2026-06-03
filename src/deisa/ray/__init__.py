from typing import Hashable

# The type used to represent an iteration.
# A Dask array is identified by its name and a timestep.
Timestep = Hashable

from .bridge import Bridge  # noqa: E402
from .window_handler import Deisa  # noqa: E402

__all__ = ["Bridge", "Deisa", "Timestep"]
