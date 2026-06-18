import pytest

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


# TODO remove when memory handling is done from bridge size checking that
# ray.put can happen because enough memory is available.
def test_max_simulation_ahead_is_read_from_kwargs():
    d = Deisa(max_simulation_ahead=2)
    assert d.max_simulation_ahead == 2


def test_unexpected_init_kwarg_is_rejected():
    with pytest.raises(TypeError, match="unexpected keyword argument 'unknown'"):
        Deisa(unknown=True)
