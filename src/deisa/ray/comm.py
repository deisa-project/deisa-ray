from deisa.core import ICommunicator

try:
    from mpi4py import MPI as _MPI
except ImportError:
    _MPI = None


def normalize_comm(comm) -> ICommunicator | None:
    """
    Normalize supported communicator inputs to the DEISA communicator protocol.

    Parameters
    ----------
    comm : deisa.core.ICommunicator or mpi4py.MPI.Comm or None
        Existing DEISA communicator, raw MPI communicator, or ``None``.

    Returns
    -------
    deisa.core.ICommunicator or None
        ``None`` if no communicator was provided, the original communicator
        unchanged, or an :class:`MPICommAdapter` wrapping a raw ``mpi4py``
        communicator.
    """
    if comm is None:
        return None

    if _MPI is not None and isinstance(comm, _MPI.Comm):
        return MPICommAdapter(comm)

    return comm


class MPICommAdapter(ICommunicator):
    """Adapter exposing an MPI communicator via the shared Comm protocol."""

    def __init__(self, comm):
        """
        Wrap an MPI communicator.

        Parameters
        ----------
        comm : mpi4py.MPI.Comm
            MPI communicator to wrap.
        """
        self._comm = comm
        self.rank = comm.Get_rank()
        self.world_size = comm.Get_size()

    def Get_rank(self) -> int:
        """Return this communicator rank."""
        return self.rank

    def Get_size(self) -> int:
        """Return this communicator world size."""
        return self.world_size

    def gather(self, data, root: int = 0):
        """Gather Python objects to ``root``."""
        return self._comm.gather(data, root=root)

    def barrier(self) -> None:
        """Block until all MPI ranks reach this barrier."""
        self._comm.Barrier()

    def bcast(self, obj, root: int = 0):
        """Broadcast a Python object from ``root`` to all MPI ranks."""
        return self._comm.bcast(obj, root=root)


class NoOpComm(ICommunicator):
    """Fallback communicator that no-ops synchronization calls."""

    def __init__(self, rank: int = 0, world_size: int = 1):
        """
        Create a dummy communicator for single-process environments.

        Parameters
        ----------
        rank : int, optional
            Rank to report. Defaults to 0.
        world_size : int, optional
            World size to report. Defaults to 1.
        """
        self.rank = rank
        self.world_size = world_size

    def Get_rank(self) -> int:
        """Return this communicator rank."""
        return self.rank

    def Get_size(self) -> int:
        """Return this communicator world size."""
        return self.world_size

    def gather(self, data, root: int = 0):
        """Gather Python objects to ``root``."""
        return [data] if self.rank == root else None

    def barrier(self) -> None:
        """No-op barrier for single-process setups."""
        return

    def bcast(self, obj, root: int = 0):
        """Return ``obj`` unchanged in single-process setups."""
        return obj
