import datetime

import torch.distributed as dist
from deisa.core import ICommunicator


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


# TODO: Add test about comm size > declared world size.
def init_gloo_comm(
    world_size: int, rank: int, master_addr: str = "127.0.0.1", master_port: int = 29500, timeout_s: int = 120
) -> ICommunicator:
    """
    Set up a Gloo communicator backed by a TCP store for tests.

    This helper intentionally lives in tests because torch is a dev dependency,
    not a runtime dependency of deisa-ray.
    """
    timeout = datetime.timedelta(seconds=timeout_s)

    store = dist.TCPStore(
        host_name=master_addr,
        port=master_port,
        world_size=world_size,
        is_master=(rank == 0),
        timeout=timeout,
        wait_for_workers=True,
    )

    dist.init_process_group(
        backend="gloo",
        store=store,
        world_size=world_size,
        rank=rank,
        timeout=timeout,
    )
    return TorchDistComm(rank=rank, world_size=world_size)


class TorchDistComm(ICommunicator):
    """Torch distributed communicator implementing the Comm protocol for tests."""

    def __init__(self, *, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size

    def Get_rank(self) -> int:
        return self.rank

    def Get_size(self) -> int:
        return self.world_size

    def gather(self, data, root: int = 0):
        gathered = [None for _ in range(self.world_size)] if self.rank == root else None
        dist.gather_object(data, gathered, dst=root)
        return gathered

    def barrier(self) -> None:
        dist.barrier()

    def bcast(self, obj, root: int = 0):
        objects = [obj]
        dist.broadcast_object_list(objects, src=root)
        return objects[0]
