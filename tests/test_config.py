# tests/test_config.py
import pytest

import deisa.ray as deisa
from deisa.ray.window_handler import Deisa
from deisa.ray.config import ConfigError


@pytest.fixture(autouse=True)
def reset_deisa_config():
    # Ensure every test starts from defaults and unlocked state.
    deisa.config._reset_for_tests()
    yield
    deisa.config._reset_for_tests()


def test_default_flag_is_false():
    assert deisa.config.experimental_distributed_scheduling_enabled is False


def test_enable_with_default_argument():
    deisa.config.enable_experimental_distributed_scheduling()
    assert deisa.config.experimental_distributed_scheduling_enabled is True


def test_disable_explicitly():
    deisa.config.enable_experimental_distributed_scheduling(True)
    deisa.config.enable_experimental_distributed_scheduling(False)
    assert deisa.config.experimental_distributed_scheduling_enabled is False


def test_non_bool_rejected():
    with pytest.raises(TypeError):
        deisa.config.enable_experimental_distributed_scheduling("yes")  # type: ignore[arg-type]


def test_config_locks_on_instantiation():
    assert deisa.config.is_locked() is False
    _ = Deisa(n_sim_nodes=0)
    assert deisa.config.is_locked() is True


def test_cannot_mutate_after_instantiation():
    _ = Deisa(n_sim_nodes=0)
    with pytest.raises(ConfigError):
        deisa.config.enable_experimental_distributed_scheduling(True)


def test_instance_copies_value_at_construction_time():
    deisa.config.enable_experimental_distributed_scheduling(True)
    d = Deisa(n_sim_nodes=0)
    assert d._experimental_distributed_scheduling_enabled is True
