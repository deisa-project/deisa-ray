from typing import Protocol
import torch.distributed as dist
import datetime


def init_gloo_comm(
    world_size: int, rank: int, master_addr: str = "127.0.0.1", master_port: int = 29500, timeout_s: int = 120
):
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


class Comm(Protocol):
    rank: int
    world_size: int

    def barrier(self) -> None: ...


class MPICommAdapter:
    def __init__(self, comm):
        self._comm = comm
        self.rank = comm.Get_rank()
        self.world_size = comm.Get_size()

    def barrier(self) -> None:
        self._comm.Barrier()


class TorchDistComm:
    def __init__(self, *, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size

    def barrier(self) -> None:
        dist.barrier()


class NoOpComm:
    def __init__(self, rank: int = 0, world_size: int = 1):
        self.rank = rank
        self.world_size = world_size

    def barrier(self) -> None:
        return
