import asyncio
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class AsyncDict(Generic[K, V]):
    """
    Minimal async-friendly dictionary with event-based waiting for new keys.

    The class mirrors the subset of the standard mapping interface that is
    needed by the Ray actors. Writes trigger an ``asyncio.Event`` so coroutines
    can wait until a key appears without busy-waiting.
    """

    def __init__(self) -> None:
        """Initialize an empty dictionary and its notification event."""
        self._data: dict[K, V] = {}
        self._new_key_event = asyncio.Event()

    def keys(self) -> list[K]:
        """
        Return the stored keys.

        Returns
        -------
        dict_keys
            Dynamic view of the keys currently stored in the dictionary.
        """
        return self._data.keys()

    def __setitem__(self, key: K, value: V) -> None:
        """
        Store ``value`` under ``key`` and wake any waiters.

        Parameters
        ----------
        key : K
            Key to insert or update.
        value : V
            Value associated with ``key``.
        """
        self._data[key] = value
        self._new_key_event.set()
        self._new_key_event.clear()

    def __getitem__(self, key: K) -> V:
        """
        Return the value stored under ``key``.

        Parameters
        ----------
        key : K
            Key to look up.

        Returns
        -------
        V
            Value associated with ``key``.

        Raises
        ------
        KeyError
            If ``key`` is not present.
        """
        return self._data[key]

    async def wait_for_key(self, key: K) -> V:
        """
        Block until ``key`` exists and return its value.

        Parameters
        ----------
        key : K
            Key to wait for.

        Returns
        -------
        V
            The value stored under ``key`` once it becomes available.
        """
        while key not in self._data:
            await self._new_key_event.wait()
        return self._data[key]

    def __contains__(self, key: K) -> bool:
        """
        Return whether ``key`` is present.

        Parameters
        ----------
        key : K
            Key to test.

        Returns
        -------
        bool
            ``True`` when ``key`` exists in the dictionary.
        """
        return key in self._data

    def __len__(self) -> int:
        """
        Return the number of stored items.

        Returns
        -------
        int
            Count of key-value pairs currently stored.
        """
        return len(self._data)
