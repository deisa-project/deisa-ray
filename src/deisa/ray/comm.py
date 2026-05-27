import datetime

import torch.distributed as dist
from deisa.core import ICommunicator

try:
    from mpi4py import MPI as _MPI
except ImportError:
    _MPI = None


# TODO : Add test about comm size > declared wolrd size
def init_gloo_comm(
    world_size: int, rank: int, master_addr: str = "127.0.0.1", master_port: int = 29500, timeout_s: int = 120
) -> ICommunicator:
    """
    Set up a Gloo communicator backed by a TCP store.

    Parameters
    ----------
    world_size : int
        Number of ranks participating in the communicator.
    rank : int
        Rank ID of the current process.
    master_addr : str, optional
        Hostname or IP address of the master rendezvous node. Defaults to
        ``"127.0.0.1"``.
    master_port : int, optional
        Port of the master rendezvous node. Defaults to 29500.
    timeout_s : int, optional
        Timeout (seconds) for rendezvous setup. Defaults to 120.

    Returns
    -------
    TorchDistComm
        Wrapper around the initialized PyTorch process group.
    """
    timeout = datetime.timedelta(seconds=timeout_s)

    # Rank 0 hosts the rendezvous store; everyone else connects.
    store = dist.TCPStore(
        host_name=master_addr,
        port=master_port,
        world_size=world_size,
        is_master=(rank == 0),
        timeout=timeout,
        wait_for_workers=True,  # optional; OK to leave default
    )

    dist.init_process_group(
        backend="gloo",
        store=store,
        world_size=world_size,
        rank=rank,
        timeout=timeout,
    )
    return TorchDistComm(rank=rank, world_size=world_size)


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

    def broadcast_object(self, obj, src: int = 0):
        """Broadcast a Python object from ``src`` to all MPI ranks."""
        return self.bcast(obj, root=src)


class TorchDistComm(ICommunicator):
    """Torch distributed communicator implementing the Comm protocol."""

    def __init__(self, *, rank: int, world_size: int):
        """
        Initialize metadata for a torch distributed communicator.

        Parameters
        ----------
        rank : int
            Rank of the current process.
        world_size : int
            Total number of ranks in the communicator.
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
        gathered = [None for _ in range(self.world_size)] if self.rank == root else None
        dist.gather_object(data, gathered, dst=root)
        return gathered

    def barrier(self) -> None:
        """Block until all Torch distributed ranks reach this barrier."""
        dist.barrier()

    def bcast(self, obj, root: int = 0):
        """Broadcast a Python object from ``root`` to all Torch distributed ranks."""
        objects = [obj]
        dist.broadcast_object_list(objects, src=root)
        return objects[0]

    def broadcast_object(self, obj, src: int = 0):
        """Broadcast a Python object from ``src`` to all Torch distributed ranks."""
        return self.bcast(obj, root=src)


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

    def broadcast_object(self, obj, src: int = 0):
        """Return ``obj`` unchanged in single-process setups."""
        return self.bcast(obj, root=src)
