"""Regression tests for per-timestep callback de-duplication.

These tests exercise ``Deisa._inspect_callbacks`` directly (no Ray cluster):
they drive ``has_new_timestep`` / ``has_seen_array`` by hand and assert that a
callback is not invoked twice for the same, already-consumed update.

Background: ``execute_callbacks`` calls ``_inspect_callbacks`` once per incoming
array within a timestep, while ``has_new_timestep`` is only cleared when the
timestep advances. Without per-callback bookkeeping, a callback whose own arrays
had already been handled would be re-invoked every time an unrelated array of
the same timestep arrived.
"""

from collections import deque

from deisa.ray.types import Window, _CallbackConfig
from deisa.ray.window_handler import Deisa


def _make_deisa(names):
    d = Deisa()
    # _determine_callback_args reads from queue_per_array; seed one element.
    for name in names:
        d.queue_per_array[name] = deque([object()], maxlen=1)
    return d


def _register(d, names, when):
    calls = []

    def callback(**kwargs):
        calls.append(kwargs)

    d.registered_callbacks.append(
        _CallbackConfig(
            simulation_callback=callback,
            arrays_description=[Window(n) for n in names],
            exception_handler=lambda exc: None,
            when=when,
        )
    )
    return calls


def test_and_single_array_fires_once_per_timestep():
    """A single-array AND callback must not re-fire for every other array
    streamed on the same timestep."""
    d = _make_deisa(["a"])
    calls = _register(d, ["a"], "AND")

    d.has_seen_array["a"] = True
    d.has_new_timestep["a"] = True

    for _ in range(8):  # 8 inspections on the same timestep
        d._inspect_callbacks(10)
    assert len(calls) == 1

    # A newer timestep allows it to fire again.
    d.has_new_timestep["a"] = True
    d._inspect_callbacks(11)
    assert len(calls) == 2


def test_and_multi_array_fires_once_when_all_updated():
    d = _make_deisa(["a", "b"])
    calls = _register(d, ["a", "b"], "AND")

    d.has_seen_array["a"] = d.has_seen_array["b"] = True
    d.has_new_timestep["a"] = True
    d._inspect_callbacks(5)  # only "a" is new -> no call
    assert len(calls) == 0

    d.has_new_timestep["b"] = True
    for _ in range(5):
        d._inspect_callbacks(5)  # all new -> exactly one call
    assert len(calls) == 1


def test_or_fires_once_per_updated_array_not_once_per_inspection():
    d = _make_deisa(["a", "b"])
    calls = _register(d, ["a", "b"], "OR")

    d.has_seen_array["a"] = d.has_seen_array["b"] = True

    d.has_new_timestep["a"] = True
    d._inspect_callbacks(3)  # "a" updated -> one call
    assert len(calls) == 1
    for _ in range(5):
        d._inspect_callbacks(3)  # nothing new -> no extra call
    assert len(calls) == 1

    d.has_new_timestep["b"] = True
    d._inspect_callbacks(3)  # "b" updated -> one call
    assert len(calls) == 2
    for _ in range(5):
        d._inspect_callbacks(3)
    assert len(calls) == 2


def test_or_warmup_batch_collapses_into_one_call():
    """Before every array has been seen at least once, updates that are already
    pending are consumed together by a single call (this reproduces the
    ``2 * N - 1`` counts asserted in tests/test_when.py)."""
    d = _make_deisa(["a", "b"])
    calls = _register(d, ["a", "b"], "OR")

    d.has_new_timestep["a"] = True
    d.has_seen_array["a"] = True
    d._inspect_callbacks(0)  # "b" never seen -> suppressed, "a" not consumed
    assert len(calls) == 0

    d.has_new_timestep["b"] = True
    d.has_seen_array["b"] = True
    d._inspect_callbacks(0)  # batch {a, b} -> a single call
    assert len(calls) == 1
    for _ in range(5):
        d._inspect_callbacks(0)
    assert len(calls) == 1


def test_callbacks_do_not_consume_each_others_updates():
    """Two callbacks sharing an array each get their own update view."""
    d = _make_deisa(["a"])
    d.has_seen_array["a"] = True
    and_calls = _register(d, ["a"], "AND")
    or_calls = _register(d, ["a"], "OR")

    d.has_new_timestep["a"] = True
    d._inspect_callbacks(0)

    assert len(and_calls) == 1
    assert len(or_calls) == 1
