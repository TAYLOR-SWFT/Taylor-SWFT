from ..core.taylor_swft import Reverberator
from ..room.spatial_model import SWFTRoom
from .custom_typing import PointType, baseline_func_type
from .utils import get_ism_order
from pyroomacoustics.parameters import constants as pra_constants
from pyroomacoustics.simulation import compute_ism_rir, compute_rt_rir
from scipy.interpolate import PchipInterpolator
from torch import Tensor
from typing import cast
import numpy as np
import torch


def rir_taylor_swft(
    room: SWFTRoom,
    source_pos: PointType,
    mic_pos: PointType,
    **kwargs,
) -> Tensor:
    """Compute RIR using Taylor-SWFT method.

    Uses the Taylor expansion-based SWFT method to efficiently compute room
    impulse responses by modal decomposition.

    Args:
        room: SWFT room object.
        source_pos: Source position [x, y, z] in meters.
        mic_pos: Microphone position [x, y, z] in meters.
        **kwargs: Additional arguments including:
            - wanted_sources (int): Limit on number of image sources. Defaults to 1e6.

    Returns:
        Tensor: Room impulse response waveform.
    """
    tswft = Reverberator(room)
    o = get_ism_order(len(room.room.walls), kwargs.get("wanted_sources", int(1e6)))
    return tswft.get_rir_at_point(point=mic_pos, source_point=source_pos, order=o)


def rir_ism(
    room: SWFTRoom,
    source_pos: PointType,
    mic_pos: PointType,
    **kwargs,
) -> Tensor:
    """Compute RIR using Image Source Model (ISM).

    Uses pyroomacoustics' ISM implementation for reference RIR computation.

    Args:
        room: SWFT room object.
        source_pos: Source position [x, y, z] in meters.
        mic_pos: Microphone position [x, y, z] in meters.
        **kwargs: Additional arguments including:
            - wanted_sources (int): Limit on number of image sources. Defaults to 1e9.

    Returns:
        Tensor: Room impulse response waveform computed via ISM.
    """
    # Set the microphone position
    if room.room.mic_array is None:
        room.room.add_microphone(mic_pos)
    else:
        room.room.mic_array.R[:, 0] = mic_pos

    # Set the source position
    if len(room.room.sources) == 0:
        room.room.add_source(source_pos)
    else:
        room.room.sources[0].position = source_pos

    o = get_ism_order(len(room.room.walls), kwargs.get("wanted_sources", int(1e9)))
    room.room.max_order = o
    room.room._update_room_engine_params()
    room.room.image_source_model()
    visibility = cast(list[np.ndarray], room.room.visibility)
    ir_ism = compute_ism_rir(
        room.room.sources[0],
        mic_pos,
        None,
        visibility[0][0, :],
        pra_constants.get("frac_delay_length"),
        room.room.c,
        room.room.fs,
        room.room.octave_bands,
        air_abs_coeffs=room.room.air_absorption,
        min_phase=room.room.min_phase,
    )
    return torch.from_numpy(ir_ism.copy())


def rir_rt(
    room: SWFTRoom,
    source_pos: PointType,
    mic_pos: PointType,
    **kwargs,
) -> Tensor:
    """Compute RIR using Ray Tracing method.

    Uses pyroomacoustics' ray tracing algorithm for late reflections estimation.

    Args:
        room: SWFT room object.
        source_pos: Source position [x, y, z] in meters.
        mic_pos: Microphone position [x, y, z] in meters.
        **kwargs: Additional arguments including:
            - n_rays (int): Number of rays to trace. Defaults to 1000.
            - receiver_radius (float): Receiver capture radius in meters. Defaults to 0.5.

    Returns:
        Tensor: Room impulse response computed via ray tracing.
    """
    # Set the microphone position
    if room.room.mic_array is None:
        room.room.add_microphone(mic_pos)
    else:
        room.room.mic_array.R[:, 0] = mic_pos

    # Set the source position
    if len(room.room.sources) == 0:
        room.room.add_source(source_pos)
    else:
        room.room.sources[0].position = source_pos

    room.room._set_ray_tracing_options(
        use_ray_tracing=True,
        n_rays=kwargs.get("n_rays", int(1000)),
        receiver_radius=kwargs.get("receiver_radius", 0.5),
    )
    room.room.ray_tracing()
    ir_rt = compute_rt_rir(
        room.room.rt_histograms[0][0],
        room.room.rt_args["hist_bin_size"],
        room.room.rt_args["hist_bin_size_samples"],
        room.room.get_volume(),
        pra_constants.get("frac_delay_length"),
        room.room.c,
        room.room.fs,
        room.room.octave_bands,
        air_abs_coeffs=room.room.air_absorption,
    )
    return torch.from_numpy(ir_rt.copy())


def rir_ism_rt(
    room: SWFTRoom,
    source_pos: PointType,
    mic_pos: PointType,
    **kwargs,
) -> Tensor:
    """Compute RIR using combined ISM and Ray Tracing method.

    Computes early reflections via ISM and late reflections via ray tracing,
    then combines them into a single RIR.

    Args:
        room: SWFT room object.
        source_pos: Source position [x, y, z] in meters.
        mic_pos: Microphone position [x, y, z] in meters.
        **kwargs: Passed to both rir_ism and rir_rt functions.

    Returns:
        Tensor: Combined RIR waveform.
    """
    ir_ism = rir_ism(room, source_pos, mic_pos, **kwargs)
    ir_rt = rir_rt(room, source_pos, mic_pos, **kwargs)
    max_len = max(ir_ism.shape[-1], ir_rt.shape[-1])
    ir_total = torch.zeros(max_len, dtype=ir_ism.dtype, device=ir_ism.device)
    ir_total[: ir_ism.shape[-1]] += ir_ism
    ir_total[: ir_rt.shape[-1]] += ir_rt
    return ir_total


def rir_noise(
    room: SWFTRoom,
    source_pos: PointType,
    mic_pos: PointType,
    **kwargs,
) -> Tensor:
    """Generate synthetic noise RIR using RT60 mean.

    Creates an exponentially decaying noise signal based on the room's
    reverberation time profile as a baseline/reference.

    Args:
        room: SWFT room object.
        source_pos: Source position (unused for noise baseline).
        mic_pos: Microphone position (unused for noise baseline).
        **kwargs: Unused, for API compatibility.

    Returns:
        Tensor: Synthetic exponentially decaying noise waveform.
    """
    rt_octave_profile = room.get_rt60_profile()
    lin_freqs = torch.linspace(0, room.center_freqs[-1], 512)
    rt_linear = PchipInterpolator(room.center_freqs, rt_octave_profile)(lin_freqs)
    rt_mean = rt_linear.mean()
    _3ln10 = 3 * torch.log(torch.tensor(10.0))
    a = _3ln10 / (room.room.fs * rt_mean)
    L = int(2 * room.room.fs * rt_mean)
    e = torch.exp(-a * torch.arange(L))
    return e * torch.randn_like(e)


ALL_BASELINES: dict[str, baseline_func_type] = {
    "ism_rt": rir_ism_rt,
    "ism": rir_ism,
    "noise": rir_noise,
    "rt": rir_rt,
    "taylor_swft": rir_taylor_swft,
}

BASELINE_KWARGS = {
    "ism_rt": {"wanted_sources": int(1e6), "n_rays": 5000, "receiver_radius": 0.5},
    "ism": {"wanted_sources": int(1e10)},
    "noise": {},
    "rt": {"n_rays": 5000, "receiver_radius": 0.5},
    "taylor_swft": {"wanted_sources": int(1e6)},
}
