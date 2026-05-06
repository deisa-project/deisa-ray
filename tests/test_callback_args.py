from deisa.ray.types import Window
from deisa.ray.window_handler import Deisa


def test_register_accepts_callback_arg_combinations(monkeypatch):
    monkeypatch.setattr(Deisa, "_ensure_connected", lambda self: None)
    deisa = Deisa()

    @deisa.register("arr1")
    def cb_string_default(arr1):
        pass

    @deisa.register("arr1", "arr2")
    def cb_two_strings_default(arr1, arr2):
        pass

    @deisa.register(Window("arr1"))
    def cb_window_default(arr1):
        pass

    @deisa.register(Window("arr1", 2))
    def cb_window_size(arr1):
        pass

    @deisa.register(Window("arr1", 2), Window("arr2", 5))
    def cb_two_window_sizes(arr1, arr2):
        pass

    @deisa.register(Window("arr1", 2), Window("arr2", 5), "arr3")
    def cb_mixed_args(arr1, arr2, arr3):
        pass

    registered_arrays = [
        [(window.name, window.window_size) for window in cfg.arrays_description]
        for cfg in deisa.registered_callbacks
    ]

    assert registered_arrays == [
        [("arr1", 1)],
        [("arr1", 1), ("arr2", 1)],
        [("arr1", 1)],
        [("arr1", 2)],
        [("arr1", 2), ("arr2", 5)],
        [("arr1", 2), ("arr2", 5), ("arr3", 1)],
    ]


def test_register_callback_accepts_unpacked_window_list(monkeypatch):
    monkeypatch.setattr(Deisa, "_ensure_connected", lambda self: None)
    deisa = Deisa()

    def callback(arr1, arr2):
        pass

    windows = [Window("arr1", 2), Window("arr2", 5)]
    assert deisa.register_callback(callback, *windows) is callback

    registered_windows = deisa.registered_callbacks[0].arrays_description
    assert registered_windows == windows
