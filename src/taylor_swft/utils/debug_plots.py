from ..room.datasets import BRASBenchmarkToSWFTRoom, BRASItemSWFTRoom
from .baselines import rir_taylor_swft, rir_noise, rir_ism, rir_rt
from .custom_typing import baseline_func_type
from pyroomacoustics.parameters import constants
from scipy.interpolate import PchipInterpolator
from taylor_swft.room.spatial_model import make_demo_room, SWFTRoom
from taylor_swft.synthesis.rir_synthesizer import PMatrix
from torch import Tensor
from torchaudio.functional import convolve
from torchaudio.transforms import MelSpectrogram
from typing import cast
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_rir_func(item: BRASItemSWFTRoom, rir_func: baseline_func_type) -> None:
    """Plot computed and reference RIRs in time domain.

    Generates a matplotlib figure comparing the RIR computed by a baseline method
    with the reference (measured) RIR for a given scene.

    Args:
        item: BRAS dataset item with room and position data.
        rir_func: Baseline RIR computation function.
    """
    print(f"Plotting RIR for {rir_func.__name__}...")
    rir = rir_func(
        item["swft_room"],
        item["source_position"],
        item["receiver_position"],
    )
    ref = item["waveform"].squeeze()
    fs = item["swft_room"].room.fs
    L = max(len(rir), len(ref))
    time = torch.arange(L) / fs

    plt.figure()
    plt.plot(time[: len(rir)], rir, label="RIR")
    plt.plot(time[: len(ref)], ref, label="Reference")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"RIR with {rir_func.__name__}")
    plt.legend()


def show_baselines() -> None:
    """Demonstrate all baseline RIR computation methods on BRAS data.

    Loads a dataset sample and plots RIRs computed by taylor_swft, noise,
    ism, and rt baseline methods for visual comparison.
    """
    constants.set("octave_bands_base_freq", 31.25)
    swft_dataset = BRASBenchmarkToSWFTRoom(
        fs=32000,
        ignore_keys=["CR1"],
        material_types="initial",
    )

    item = swft_dataset[0]
    plot_rir_func(item, rir_taylor_swft)
    plot_rir_func(item, rir_noise)
    plot_rir_func(item, rir_ism)
    plot_rir_func(item, rir_rt)
    plt.show()


def get_ir_from_modes(modes_at_begin: Tensor) -> Tensor:
    """Reconstruct impulse response from frequency-domain modes.

    Performs inverse FFT and applies spectral shaping via cepstral processing
    to convert octave-band mode magnitudes back to time-domain impulse response.

    Args:
        modes_at_begin: Frequency-domain mode magnitudes at octave bands.

    Returns:
        Tensor: Time-domain impulse response reconstructed from modes.
    """
    # Resample points at the begining of each band
    cepstrum = cast(Tensor, torch.fft.irfft(torch.log(modes_at_begin)))
    l = cepstrum.shape[-1]
    cepstrum[1 : l // 2] *= 2
    cepstrum[l // 2 + 1 :] = 0
    modes_at_begin = torch.exp(torch.fft.rfft(cepstrum, 64))
    return torch.fft.irfft(modes_at_begin)


def octave_to_linear(
    octave_profile: Tensor, freqs: Tensor, n_target_freqs: int = 64
) -> tuple[np.typing.ArrayLike, Tensor]:
    """Convert an octave-band RT60 profile to a linear frequency profile by interpolation."""
    # Compute the center frequencies of the octave bands
    lin_freqs = torch.linspace(
        0,
        freqs[-1],
        n_target_freqs,
    )
    linear_profile = PchipInterpolator(freqs, octave_profile)(lin_freqs)
    return linear_profile, lin_freqs


def P_vs_P_transpose(
    visualize: bool = False,
    swft_room: SWFTRoom | None = None,
) -> None:
    """Verify mathematical equivalence of P*G and P^T*G^T operations.

    Compares covariance and mel-spectrogram statistics of RIRs generated via
    P*G*epsilon (standard processing) and P^T*G^T*epsilon (transposed operation)
    to validate correct implementation of the transpose convolution operator.

    Args:
        visualize: If True, display detailed plots and statistics. Defaults to False.
        swft_room: SWFT room to analyze. If None, uses demo room. Defaults to None.

    Raises:
        AssertionError: If error ratios exceed acceptable thresholds (-25 dB).
    """
    if swft_room is None:
        swft_room = make_demo_room(verbose=False)
        N_stats = 10
        sr = swft_room.room.fs // 10
        Lh = sr
        nfft = 64
        n_mels = 8
    else:
        swft_room = swft_room
        N_stats = 50
        sr = swft_room.room.fs
        Lh = sr * 2
        nfft = 256
        n_mels = 32

    bbox = swft_room.room.get_bbox()
    rt_octave = swft_room.get_rt60_profile()
    modes_octave = swft_room.get_frequency_response_at_point(np.mean(bbox, axis=1))

    h_rt, freqs = octave_to_linear(
        torch.tensor(rt_octave), torch.tensor(swft_room.center_freqs)
    )
    g_tf, _ = octave_to_linear(
        torch.tensor(modes_octave), torch.tensor(swft_room.center_freqs)
    )
    g_tf = torch.tensor(g_tf, dtype=torch.float32)
    g = get_ir_from_modes(g_tf).unsqueeze(0)
    M = g.shape[-1]

    p_matrix = PMatrix(
        rt_60_profile=torch.tensor(h_rt, dtype=torch.float32), sample_rate=sr
    )
    epsilon = torch.randn([N_stats, Lh], dtype=torch.float32)

    ######### GP * epsilon ###########
    rirs_P = convolve(p_matrix.taylor_mul(epsilon), g, mode="full")[..., : -M + 1]

    ########## P^T G^T * epsilon ###########
    from warnings import filterwarnings, resetwarnings

    filterwarnings("ignore", category=UserWarning)
    # Throws a warning about the convolution being inefficient.
    rirs_Pt = p_matrix.transpose_mul(
        convolve(epsilon, g.flip(dims=[-1]), mode="full")[..., M - 1 :]
    )
    resetwarnings()

    ################### Compare the covariance of the RIRs obtained from PG and G^TP^T. ################
    cov_P = torch.cov(rirs_P.T)
    cov_Pt = torch.cov(rirs_Pt.T)

    absolute_error = torch.abs(cov_P - cov_Pt).max()
    max_error_ratio_db = 20 * torch.log10(absolute_error / cov_Pt.abs().max())
    assert (
        max_error_ratio_db < -25
    ), f"Maximum error ratio between P and P^T is too high: {max_error_ratio_db.item():.2f} dB"

    ################ Compare the Mel-spectrograms of the RIRs obtained from PG and G^TP^T. ################
    mel_transform = MelSpectrogram(
        sample_rate=sr, n_fft=nfft, hop_length=nfft // 2, n_mels=n_mels
    )

    spec_P = mel_transform(rirs_P)
    spec_Pt = mel_transform(rirs_Pt)
    spec_error = torch.abs(spec_P - spec_Pt).mean(dim=0)
    max_spec_error_ratio_db = 20 * torch.log10(spec_error.max() / spec_Pt.abs().max())
    assert (
        max_spec_error_ratio_db < -25
    ), f"Maximum spectrogram error ratio between P and P^T is too high: {max_spec_error_ratio_db.item():.2f} dB"

    if visualize:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib import colors

        print(f"Maximum relative covariance error: {max_error_ratio_db.item():.4e} dB")
        print(
            f"Maximum relative spectrogram error: {max_spec_error_ratio_db.item():.4e} dB"
        )

        rir_P = rirs_P[0]
        rir_Pt = rirs_Pt[0]

        major_fontsize = 22
        global_fontsize = 18
        ticks_fontsize = 15
        plt.rcParams.update(
            {
                "font.size": global_fontsize,
                "axes.titlesize": major_fontsize,
                "axes.labelsize": global_fontsize,
                "xtick.labelsize": ticks_fontsize,
                "ytick.labelsize": ticks_fontsize,
            }
        )

        plt.figure()
        plt.plot(p_matrix.interpolated_rt.detach().cpu(), label="RT60 profile")
        plt.legend()

        plt.figure()
        plt.plot(g_tf.squeeze().detach().cpu(), label="Frequency response of g")
        plt.legend()

        plt.figure()
        plt.plot(rir_P.detach().cpu(), label="RIR from P * epsilon")
        plt.plot(rir_Pt.detach().cpu(), label="RIR from P^T * epsilon")
        plt.legend()

        plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 5, width_ratios=[1, 1, 1, 1, 0.1])
        axs1 = plt.subplot(gs[0, :2])
        axs2 = plt.subplot(gs[0, 2:4], sharey=axs1)
        axs3 = plt.subplot(gs[1, 1:3])
        axs4 = plt.subplot(gs[0:, 4])
        logmag_P = 20 * torch.log10(spec_P[0].cpu() + 1e-8)
        logmag_Pt = 20 * torch.log10(spec_Pt[0].cpu() + 1e-8)
        logmag_error = 20 * torch.log10(
            (spec_P[0].cpu() - spec_Pt[0].cpu()).abs() + 1e-8
        )
        norm = colors.Normalize(
            vmin=np.min([logmag_P.min(), logmag_Pt.min(), logmag_error.min()]),
            vmax=np.max([logmag_P.max(), logmag_Pt.max(), logmag_error.max()]),
        )
        im_P = axs1.imshow(
            logmag_P, aspect="auto", origin="lower", norm=norm, cmap="plasma"
        )
        axs1.set_title("Mel-spectrogram of $G_xP\\varepsilon$")

        im_Pt = axs2.imshow(
            logmag_Pt, aspect="auto", origin="lower", norm=norm, cmap="plasma"
        )
        axs2.set_title("Mel-spectrogram of $P^T G_x^T\\varepsilon$")

        im_error = axs3.imshow(
            logmag_error,
            aspect="auto",
            origin="lower",
            norm=norm,
            cmap="plasma",
        )

        axs3.set_title("Absolute error")

        # Set y-axis to log scale with frequency labels
        display_step = 5
        mel_freqs = torch.exp(torch.linspace(0, np.log(sr / 2), display_step))
        labels = [f"{int(f)}" for f in mel_freqs.cpu()]
        label_pos = np.linspace(0, n_mels - 1, display_step)
        axs1.set_ylabel("Frequency (Hz)")
        axs3.set_ylabel("Frequency (Hz)")
        axs1.set_yticks(label_pos, labels=labels, rotation=90)
        axs2.set_yticks(label_pos, labels=labels, rotation=90)
        axs3.set_yticks(label_pos, labels=labels, rotation=90)

        # Set x-axis to time labels
        time_step = 5
        duration_s = rir_P.shape[-1] / sr
        time_points = torch.linspace(0, duration_s, time_step)
        time_labels = [f"{t:.2f}" for t in time_points.cpu()]
        time_pos = np.linspace(0, spec_P.shape[-1] - 1, time_step)
        axs1.set_xlabel("Time (s)")
        axs2.set_xlabel("Time (s)")
        axs3.set_xlabel("Time (s)")
        axs1.set_xticks(time_pos, labels=time_labels)
        axs2.set_xticks(time_pos, labels=time_labels)
        axs3.set_xticks(time_pos, labels=time_labels)

        plt.colorbar(
            im_error,
            cax=axs4,
            label="Magnitude (dB)",
            orientation="vertical",
            fraction=0.1,
        )
        plt.tight_layout()
        plt.savefig("outputs/P_vs_Pt_spectrograms.svg")
        plt.show()
