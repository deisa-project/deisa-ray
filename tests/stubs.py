import time
import ray


@ray.remote
class StubSchedulingActor:
    """A lightweight per-node actor used by tests as a proxy."""

    def __init__(self, actor_id: int, ready_delay_s: float = 0.0):
        self.node_id = actor_id
        self._ready = False
        if ready_delay_s > 0:
            time.sleep(ready_delay_s)
        self._ready = True

    async def register_chunk_meta(self, *args, **kwargs):
        return True

    async def register_chunk(self, *args, **kwargs):
        # No-op; we don't test add_chunk here
        return True

    def ready(self):
        pass
