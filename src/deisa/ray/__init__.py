from collections.abc import Hashable

# The type used to represent an iteration.
# A Dask array is identified by its name and a timestep.
Timestep = Hashable

from .bridge import Bridge  # noqa: E402
from .mergeable_pca import (  # noqa: E402
    MergeablePCA,
    PCASummary,
    local_pca,
    local_pca_from_chunk,
    merge_pca,
    merge_tree,
)
from .window_handler import Deisa  # noqa: E402

__all__ = [
    "Bridge",
    "Deisa",
    "MergeablePCA",
    "PCASummary",
    "Timestep",
    "local_pca",
    "local_pca_from_chunk",
    "merge_pca",
    "merge_tree",
]
