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
        """Initialise configuration defaults (all features disabled and unlocked)."""
        self._locked: bool = False
        self._enable_experimental_distributed_scheduling: bool = False

    @property
    def experimental_distributed_scheduling_enabled(self) -> bool:
        """
        Read-only accessor for the experimental distributed scheduling flag.

        Returns
        -------
        bool
            ``True`` when distributed scheduling is enabled, ``False`` otherwise.
        """
        return self._enable_experimental_distributed_scheduling

    def enable_experimental_distributed_scheduling(self, enabled: bool = True) -> None:
        """
        Enable/disable experimental distributed scheduling.

        Parameters
        ----------
        enabled : bool, optional
            Desired state of the flag. Defaults to ``True``.

        Raises
        ------
        ConfigError
            If configuration has already been locked by instantiating ``Deisa``.
        TypeError
            If ``enabled`` is not a boolean.

        Examples
        --------
        >>> deisa.config.enable_experimental_distributed_scheduling()      # enable
        >>> deisa.config.enable_experimental_distributed_scheduling(False) # disable
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
        """
        Make configuration immutable for the remainder of the process.

        Notes
        -----
        Called automatically when ``Deisa()`` is instantiated. Further
        mutation attempts will raise :class:`ConfigError`.
        """
        self._locked = True

    def is_locked(self) -> bool:
        """
        Report whether the configuration has been locked.

        Returns
        -------
        bool
            ``True`` if :meth:`lock` has been called, otherwise ``False``.
        """
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
