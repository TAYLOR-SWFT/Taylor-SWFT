def test_imports():
    import taylor_swft.room.spatial_model


def test_rt60_computation():
    from taylor_swft.room.spatial_model import make_demo_room
    import numpy as np

    room_object = make_demo_room()
    rt60_profile = room_object.get_rt60_profile()
    rt60_from_eyring_formula = room_object.eyring_formula()

    assert np.allclose(rt60_profile, rt60_from_eyring_formula, rtol=1e-6)
