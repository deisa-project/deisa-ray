from __future__ import annotations

import os

from deisa.ray.errors import ConfigError

DEISA_DISTRIBUTED_SCHEDULING_ENV = "DEISA_DISTRIBUTED_SCHEDULING"


def distributed_scheduling_enabled_from_env() -> bool:
    """
    Read distributed scheduling from the process environment.

    Returns
    -------
    bool
        ``True`` when ``DEISA_DISTRIBUTED_SCHEDULING`` contains an enabled
        value, otherwise ``False``.

    Raises
    ------
    ConfigError
        If ``DEISA_DISTRIBUTED_SCHEDULING`` is set to an unsupported value.

    Notes
    -----
    ``DEISA_DISTRIBUTED_SCHEDULING`` accepts common boolean shell values:
    ``1``, ``true``, ``yes``, and ``on`` enable the feature, while unset,
    empty, ``0``, ``false``, ``no``, and ``off`` disable it.
    """
    raw_value = os.environ.get(DEISA_DISTRIBUTED_SCHEDULING_ENV)
    if raw_value is None:
        return False

    value = raw_value.strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    raise ConfigError(f"{DEISA_DISTRIBUTED_SCHEDULING_ENV} must be one of 1/0, true/false, yes/no, or on/off")
