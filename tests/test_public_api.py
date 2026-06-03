def test_deisa_ray_exports_public_entrypoints():
    from deisa.ray import Bridge, Deisa
    from deisa.ray.bridge import Bridge as BridgeFromModule
    from deisa.ray.window_handler import Deisa as DeisaFromModule

    assert Bridge is BridgeFromModule
    assert Deisa is DeisaFromModule
