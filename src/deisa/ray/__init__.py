from typing import Hashable
from .config import config as config

# The type used to represent an iteration.
# A Dask array is identified by its name and a timestep.
Timestep = Hashable
