import pytest

from deisa.ray import window_handler
from deisa.ray.window_handler import Deisa
from deisa.ray.config import (
    DEISA_DISTRIBUTED_SCHEDULING_ENV,
    ConfigError,
    distributed_scheduling_enabled_from_env,
)


@pytest.fixture(autouse=True)
def reset_deisa_config(monkeypatch):
    # Ensure every test starts from defaults and unlocked state.
    monkeypatch.delenv(DEISA_DISTRIBUTED_SCHEDULING_ENV, raising=False)
    yield
    monkeypatch.delenv(DEISA_DISTRIBUTED_SCHEDULING_ENV, raising=False)


def test_default_flag_is_false():
    assert distributed_scheduling_enabled_from_env() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_enabled_env_values(monkeypatch, value):
    monkeypatch.setenv(DEISA_DISTRIBUTED_SCHEDULING_ENV, value)
    assert distributed_scheduling_enabled_from_env() is True


@pytest.mark.parametrize("value", ["", "0", "false", "FALSE", "no", "off"])
def test_disabled_env_values(monkeypatch, value):
    monkeypatch.setenv(DEISA_DISTRIBUTED_SCHEDULING_ENV, value)
    assert distributed_scheduling_enabled_from_env() is False


def test_invalid_env_value_rejected(monkeypatch):
    monkeypatch.setenv(DEISA_DISTRIBUTED_SCHEDULING_ENV, "maybe")
    with pytest.raises(ConfigError):
        distributed_scheduling_enabled_from_env()


def test_instance_copies_env_value_at_construction_time(monkeypatch):
    monkeypatch.setenv(DEISA_DISTRIBUTED_SCHEDULING_ENV, "1")
    d = Deisa()
    assert d._experimental_distributed_scheduling_enabled is True


def test_instance_uses_construction_time_env_value(monkeypatch):
    monkeypatch.setenv(DEISA_DISTRIBUTED_SCHEDULING_ENV, "1")
    d = Deisa()
    monkeypatch.setenv(DEISA_DISTRIBUTED_SCHEDULING_ENV, "0")
    assert d._experimental_distributed_scheduling_enabled is True


def test_ray_start_is_read_from_kwargs():
    def ray_start():
        pass

    d = Deisa(ray_start=ray_start)
    assert d._ray_start is ray_start


def test_default_ray_start_retries_until_ray_initializes(monkeypatch):
    init_errors = [ConnectionError("ray runtime is not ready"), ConnectionError("still not ready")]
    init_calls = []
    sleeps = []

    monkeypatch.setattr(window_handler.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(window_handler.time, "sleep", sleeps.append)

    def fake_init(**kwargs):
        init_calls.append(kwargs)
        if init_errors:
            raise init_errors.pop(0)

    monkeypatch.setattr(window_handler.ray, "init", fake_init)

    window_handler._ray_start_impl()

    assert len(init_calls) == 3
    assert sleeps == [1.0, 1.0]
    assert init_calls[0] == {
        "address": "auto",
        "log_to_driver": False,
        "logging_level": window_handler.logging.ERROR,
    }


def test_default_ray_start_raises_after_retry_timeout(monkeypatch):
    elapsed = 0.0
    init_calls = 0

    monkeypatch.setattr(window_handler.ray, "is_initialized", lambda: False)

    def fake_monotonic():
        return elapsed

    def fake_sleep(seconds):
        nonlocal elapsed
        elapsed += seconds

    def fake_init(**kwargs):
        nonlocal init_calls
        init_calls += 1
        raise ConnectionError("ray runtime is not ready")

    monkeypatch.setattr(window_handler.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(window_handler.time, "sleep", fake_sleep)
    monkeypatch.setattr(window_handler.ray, "init", fake_init)

    with pytest.raises(ConnectionError, match="ray runtime is not ready"):
        window_handler._ray_start_impl()

    assert elapsed == 10.0
    assert init_calls == 11


def test_unexpected_init_kwarg_is_rejected():
    with pytest.raises(TypeError, match="unexpected keyword argument 'unknown'"):
        Deisa(unknown=True)
