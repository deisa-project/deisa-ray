import asyncio
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class AsyncDict(Generic[K, V]):
    def __init__(self) -> None:
        self._data: dict[K, V] = {}
        self._new_key_event = asyncio.Event()

    def __setitem__(self, key: K, value: V):
        self._data[key] = value
        self._new_key_event.set()
        self._new_key_event.clear()

    def __getitem__(self, key: K) -> V:
        return self._data[key]

    async def wait_for_key(self, key: K) -> V:
        while key not in self._data:
            await self._new_key_event.wait()
        return self._data[key]

    def __contains__(self, key: K) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)
