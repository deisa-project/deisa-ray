# deisa/config.py
from __future__ import annotations
from deisa.ray.errors import ConfigError


class Config:
    """
    Process-wide configuration for the deisa library.

    Mutable until locked. Instantiating Deisa() locks configuration to prevent
    mid-run behavior changes.
    """

    def __init__(self) -> None:
        self._locked: bool = False
        # TODO: make the default False when support for both is added
        # self._enable_experimental_distributed_scheduling: bool = False
        self._enable_experimental_distributed_scheduling: bool = True

    @property
    def experimental_distributed_scheduling_enabled(self) -> bool:
        """Read-only accessor for the current value."""
        return self._enable_experimental_distributed_scheduling

    def enable_experimental_distributed_scheduling(self, enabled: bool = True) -> None:
        """
        Enable/disable experimental distributed scheduling.

        Usage:
            deisa.config.enable_experimental_distributed_scheduling()      # enable
            deisa.config.enable_experimental_distributed_scheduling(False) # disable
        """
        if self._locked:
            raise ConfigError(
                "deisa.config is locked because Deisa() has already been instantiated. "
                "Set configuration before creating a Deisa instance."
            )
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        self._enable_experimental_distributed_scheduling = enabled

    def lock(self) -> None:
        self._locked = True

    def is_locked(self) -> bool:
        return self._locked

    # ---- Test support (explicitly non-public) ----
    def _reset_for_tests(self) -> None:
        """
        Reset config to defaults and unlock it.

        Intended for unit tests only. Do not rely on this in production.
        """
        self._locked = False
        self._enable_experimental_distributed_scheduling = False


config = Config()
