import time
import ray


def cb_identity(x):
    return x


def cb_times2(x):
    return x * 2


@ray.remote
class StubSchedulingActor:
    """A lightweight per-node actor used by tests as a proxy."""

    def __init__(self, actor_id: int, ready_delay_s: float = 0.0, callbacks=None):
        self.node_id = actor_id
        self._ready = False
        self._callbacks_ref = ray.put(callbacks or {"default": cb_identity, "double": cb_times2})
        if ready_delay_s > 0:
            time.sleep(ready_delay_s)
        self._ready = True

    def preprocessing_callbacks(self) -> ray.ObjectRef:
        return self._callbacks_ref

    async def register_chunk_meta(self, *args, **kwargs):
        return True

    async def register_chunk(self, *args, **kwargs):
        # No-op; we don't test add_chunk here
        return True

    def ready(self):
        pass
