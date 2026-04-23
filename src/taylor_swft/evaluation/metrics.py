from ..utils import constants
from dtaidistance.dtw import distance_fast as dtw
from numpy import float64
from torch import Tensor
from torch.nn.functional import pad
from torchaudio.transforms import MelSpectrogram
from typing import cast, Callable
from warnings import warn
import torch


def to_dB(x: Tensor) -> Tensor:
    """Convert tensor to decibel scale.

    Args:
        x: Tensor to convert to dB scale.

    Returns:
        Tensor converted to dB scale (10 * log10(x)).

    Warns:
        UserWarning: If any values in the tensor are non-positive, as they
            result in invalid (inf or nan) values in dB scale.
    """
    if torch.any(x <= 0):
        warn(
            "A tensor with non-positive values was passed to to_dB."
            "The result will contain invalid values."
        )

    return 10 * torch.log10(x)


def clarity_50ms(rir: Tensor, sample_rate: int) -> Tensor:
    """Calculate clarity index (C50) - energy ratio in first 50ms.

    Args:
        rir: Room impulse response tensor.
        sample_rate: Audio sample rate in Hz.

    Returns:
        Clarity index C50 in dB (energy before 50ms / energy after 50ms).

    Warns:
        UserWarning: If reverberant energy is near zero, resulting in infinite
            clarity values.
    """
    index_50ms = int(0.05 * sample_rate)

    energy_before = rir[..., :index_50ms].pow(2).sum(dim=-1)
    energy_after = rir[..., index_50ms:].pow(2).sum(dim=-1)

    if energy_after.isclose(
        torch.tensor(
            constants.ALMOST_ZERO,
            device=rir.device,
            dtype=rir.dtype,
        )
    ):
        warn("The RIR is too short to compute clarity_50ms.")
        return torch.tensor(torch.inf)

    return to_dB(energy_before / energy_after)


def definition_50ms(rir: Tensor, sample_rate: int) -> Tensor:
    """Calculate definition index (D50) - percentage of energy in first 50ms.

    Args:
        rir: Room impulse response tensor.
        sample_rate: Audio sample rate in Hz.

    Returns:
        Definition index D50 in percentage (100 * energy before 50ms / total energy).

    Warns:
        UserWarning: If RIR is shorter than 50ms.
    """
    index_50ms = int(0.05 * sample_rate)

    if index_50ms >= rir.shape[-1]:
        warn("The RIR is too short to compute definition_50ms.")
        return torch.tensor(100.0)

    energy_before = rir[..., :index_50ms].pow(2).sum(dim=-1)
    energy_total = rir.pow(2).sum(dim=-1)

    return 100 * energy_before / energy_total


def direct_to_reverberant_ratio(rir: Tensor, sample_rate: int) -> Tensor:
    """Calculate direct-to-reverberant energy ratio.

    Args:
        rir: Room impulse response tensor.
        sample_rate: Audio sample rate in Hz.

    Returns:
        Direct-to-reverberant energy ratio in dB. Direct energy is computed
        from ±5ms around the peak, reverberant energy from the rest.
    """
    t0 = rir.abs().argmax(dim=-1)
    half_win_size = int(sample_rate * constants.DRR_WIN_SIZE_S / 2)
    start = max(t0 - half_win_size, 0)
    end = t0 + half_win_size
    direct_energy = rir[..., start:end].pow(2).sum(dim=-1)
    reverberant_energy = rir[..., end:].pow(2).sum(dim=-1)

    return to_dB(direct_energy / reverberant_energy)


def reverb_time_30_dB(rir: Tensor, sample_rate: int) -> Tensor:
    """Calculate reverberation time T30 (decay from -5dB to -35dB).

    Args:
        rir: Room impulse response tensor.
        sample_rate: Audio sample rate in Hz.

    Returns:
        Reverberation time T30 in seconds.
    """
    edc_dB = to_dB(
        energy_decay_curve(rir, sample_rate).clamp(min=constants.ALMOST_ZERO)
    )
    idx_minus_5_dB = torch.argmin((edc_dB + 5).abs(), dim=-1)
    idx_minus_35_dB = torch.argmin((edc_dB + 35).abs(), dim=-1)

    return (idx_minus_35_dB - idx_minus_5_dB) / sample_rate


def reverb_time_60_dB(rir: Tensor, sample_rate: int) -> Tensor:
    return rir


def energy_decay_curve(rir: Tensor, *args) -> Tensor:
    """Calculate normalized energy decay curve (EDC).

    Args:
        rir: Room impulse response tensor.
        *args: Additional arguments (ignored, for interface compatibility).

    Returns:
        Normalized energy decay curve where EDC[t] = sum(rir[t:]^2) / sum(rir^2).
    """
    edc = rir.pow(2).flip(dims=(-1,)).cumsum(dim=-1).flip(dims=(-1,))
    return edc / edc[..., 0:1]


def energy_decay_relief(rir: Tensor, sample_rate: int) -> Tensor:
    """Calculate energy decay relief using STFT.

    Args:
        rir: Room impulse response tensor.
        sample_rate: Audio sample rate in Hz.

    Returns:
        Normalized energy decay relief computed from STFT with 30ms window.
        Shape: (frequency_bins, time_frames).
    """
    nfft = int(constants.EDR_FFT_WIN_SIZE_S * sample_rate)
    hop = nfft // 2
    window = torch.hann_window(nfft, device=rir.device, dtype=rir.dtype)
    rir_stft = torch.stft(
        rir,
        n_fft=nfft,
        hop_length=hop,
        window=window,
        return_complex=True,
    )
    power = rir_stft.real.pow(2) + rir_stft.imag.pow(2)
    edr = power.flip(dims=(-1,)).cumsum(dim=-1).flip(dims=(-1,))
    return edr / torch.sum(edr[..., 0:1], dim=-2)


def mel_energy_decay_relief(rir: Tensor, sample_rate: int) -> Tensor:
    """Calculate energy decay relief using mel-frequency spectrogram.

    Args:
        rir: Room impulse response tensor.
        sample_rate: Audio sample rate in Hz.

    Returns:
        Normalized energy decay relief computed using mel-spectrogram with
        40 mel bins and 30ms window. Shape: (mel_bins, time_frames).
    """
    nfft = int(constants.EDR_FFT_WIN_SIZE_S * sample_rate)
    hop = nfft // 2
    mel_spec_transform = MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=nfft,
        hop_length=hop,
        window_fn=torch.hann_window,
        n_mels=40,
    ).to(rir.device)
    mel_power = cast(torch.Tensor, mel_spec_transform(rir.to(torch.float32))).to(
        rir.dtype
    )
    edr = mel_power.flip(dims=(-1,)).cumsum(dim=-1).flip(dims=(-1,))
    return edr / torch.sum(edr[..., 0:1], dim=-2)


def float_distance(x: Tensor, y: Tensor) -> float:
    """Calculate mean absolute difference between two tensors.

    Args:
        x: Reference tensor.
        y: Tensor to compare.

    Returns:
        Mean absolute difference as a scalar float value.
    """
    return (x - y).abs().mean().item()


def log_distance(x: Tensor, y: Tensor) -> float:
    """Calculate mean absolute difference in log-scale (dB).

    Args:
        x: Reference tensor.
        y: Tensor to compare.

    Returns:
        Mean absolute difference in dB scale between the two tensors,
        computed after converting to dB with clipping at ALMOST_ZERO.
    """
    return (
        (
            to_dB(x.clamp(min=constants.ALMOST_ZERO))
            - to_dB(y.clamp(min=constants.ALMOST_ZERO))
        )
        .abs()
        .mean()
        .item()
    )


def dtw_distance(x: Tensor, y: Tensor) -> float:
    """Calculate Dynamic Time Warping distance between two tensors.

    Args:
        x: Reference tensor.
        y: Tensor to compare.

    Returns:
        DTW distance using a window constraint of 160 samples (5ms at 32kHz).
    """
    # 5 ms at 32 kHz
    win_len = 160
    return dtw(x.numpy().astype(float64), y.numpy().astype(float64), window=win_len)


def evaluate(rir: Tensor, sample_rate: int) -> dict[str, Tensor]:
    """Evaluate all acoustic metrics for a room impulse response.

    Args:
        rir: Room impulse response tensor.
        sample_rate: Audio sample rate in Hz.

    Returns:
        Dictionary mapping metric names to computed tensor values.
    """
    results = {}
    for name, metric in ALL_METRICS.items():
        results[name] = metric(rir, sample_rate)
    return results


def distance(rir_ref: Tensor, rir_other: Tensor, sample_rate: int) -> dict[str, Tensor]:
    """Calculate distances between metrics of two RIRs.

    Args:
        rir_ref: Reference room impulse response tensor.
        rir_other: RIR to compare against reference.
        sample_rate: Audio sample rate in Hz.

    Returns:
        Dictionary mapping metric names to distance values computed with
        appropriate distance functions (float, log, or DTW distance).
    """

    # Pad the shorter RIR
    pad_other = rir_ref.shape[-1] - rir_other.shape[-1]
    rir_other_padded = pad(rir_other / rir_other.abs().max(), (0, pad_other))

    metrics_ref = evaluate(rir_ref / rir_ref.abs().max(), sample_rate)
    metrics_other = evaluate(rir_other_padded, sample_rate)

    distances = {}
    for name, function in DISTANCES.items():
        distances[name] = function(metrics_ref[name], metrics_other[name])

    return distances


ALL_METRICS: dict[str, Callable] = {
    "clarity_50ms": clarity_50ms,
    "definition_50ms": definition_50ms,
    "direct_to_reverberant_ratio": direct_to_reverberant_ratio,
    "dtw": lambda x, sr: x,
    "energy_decay_curve": energy_decay_curve,
    "energy_decay_relief": energy_decay_relief,
    "mel_energy_decay_relief": mel_energy_decay_relief,
    "reverb_time_30_dB": reverb_time_30_dB,
}

METRICS_UNITS: dict[str, str] = {
    "clarity_50ms": "dB",
    "definition_50ms": "%",
    "direct_to_reverberant_ratio": "dB",
    "dtw": "",
    "energy_decay_curve": "",
    "energy_decay_relief": "",
    "mel_energy_decay_relief": "",
    "reverb_time_30_dB": "s",
}

DISTANCES: dict[str, Callable] = {
    "clarity_50ms": float_distance,
    "definition_50ms": float_distance,
    "direct_to_reverberant_ratio": float_distance,
    "dtw": dtw_distance,
    "energy_decay_curve": log_distance,
    "energy_decay_relief": log_distance,
    "mel_energy_decay_relief": log_distance,
    "reverb_time_30_dB": float_distance,
}

for k in DISTANCES.keys():
    assert k in ALL_METRICS, f"{k} is in DISTANCES but not in ALL_METRICS"
