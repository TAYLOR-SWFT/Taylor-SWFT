def test_imports():
    import taylor_swft.core.taylor_swft


def test_sanity_check():
    from numpy import array
    from taylor_swft.core.taylor_swft import Reverberator
    from taylor_swft.room.spatial_model import make_demo_room
    from torch import randn

    swft_room = make_demo_room()
    random_point = array([20, 10, 1.5])

    assert swft_room.room.is_inside(
        random_point
    ), "The random point should be inside the room."

    print("Loading the source audio")
    source = randn(swft_room.room.fs * 5)
    print("Creating the Reverberator object")
    taylor_swft = Reverberator(swft_room)
    print("Applying reverb at the random point")
    wet = taylor_swft.apply_reverb_at_point(source, random_point)

    assert wet.dtype == source.dtype
