from ..room.spatial_model import SWFTRoom
from ..synthesis.rir_synthesizer import PMatrix
from ..utils import constants
from ..utils.custom_typing import PointType
from pyroomacoustics.parameters import constants as pra_constants
from pyroomacoustics.simulation import compute_ism_rir
from pyroomacoustics.soundsource import SoundSource
from scipy.interpolate import PchipInterpolator
from time import perf_counter
from torch import Tensor
from torchaudio.functional import convolve
from typing import cast
import numpy as np
import torch


class Reverberator:
    room: SWFTRoom
    late_rir_operator: PMatrix
    room_sr: int
    center_freqs: np.ndarray
    late_rir_rt: Tensor

    def __init__(self, room: SWFTRoom) -> None:
        """Initialize the Reverberator with a given room model.

        Computes the late reverberation impulse response during initialization
        using the room's RT60 profile.

        Args:
            room: The SWFT room model containing geometry and acoustic properties.
        """
        self.room = room
        self.sr = room.room.fs
        self.center_freqs = room.center_freqs
        self.compute_late_rir()

    def compute_late_rir(self) -> None:
        """Compute the late reverberation impulse response using Taylor expansion.

        Interpolates the room's RT60 profile to a linear frequency axis and uses
        the PMatrix operator to synthesize the late RIR.

        Returns:
            None. Updates instance attributes: rt_profile, late_rir_operator,
            Lh (length), and late_rir (impulse response).
        """
        if hasattr(self, "late_rir"):
            return
        # Interpolate the RT60 profile to have a linear frequency axis
        self.rt_octave_profile = self.room.get_rt60_profile()
        lin_freqs = torch.linspace(
            0,
            self.center_freqs[-1],
            constants.N_LINEAR_FREQS_INTERPOLATION,
        )
        rt_linear = PchipInterpolator(self.center_freqs, self.rt_octave_profile)(
            lin_freqs
        )
        # Global attribute of the room
        self.rt_profile = torch.from_numpy(rt_linear)
        self.late_rir_operator = PMatrix(self.rt_profile, self.sr)
        self.Lh = int(self.rt_profile.max() * self.sr)
        noise = torch.randn(self.Lh)
        self.late_rir = self.late_rir_operator.taylor_mul(noise)

    def compute_late_rir_rt(
        self,
        buffer_size: int,
        context_len: int,
        device: torch.device,
    ) -> int:
        """Reshape late RIR for real-time processing with fixed buffer sizes.

        Partitions the late RIR into buffers and reverses for efficient
        convolution in real-time applications. Moves result to specified device.

        Args:
            buffer_size: Number of samples per audio buffer.
            context_len: Number of buffers to retain in the context.
            device: PyTorch device (cpu or cuda) for computation.

        Returns:
            Number of buffers in the reshaped late_rir_rt tensor.
        """
        if hasattr(self, "late_rir_rt"):
            return self.late_rir_rt.shape[0]

        rest = self.Lh % buffer_size
        n_buffers = self.Lh // buffer_size

        self.late_rir_rt = (
            self.late_rir[:-rest]
            .view(n_buffers, buffer_size)[:context_len]
            .flip(0)
            .to(device)
        )

        return self.late_rir_rt.shape[0]

    def compute_full_rir_at_point_rt(
        self,
        point: PointType,
        source_point: PointType,
        buffer_size: int,
        context_len: int,
        device: torch.device = torch.device("cpu"),
    ) -> int:
        """Compute full RIR (early + late) at a point optimized for real-time.

        Generates the early reflections using ISM, blends them with late reverberation
        using a cosine crossfade, and reshapes for efficient real-time buffer-based
        convolution.

        Args:
            point: 3D receiver position (x, y, z) in meters.
            source_point: 3D source position (x, y, z) in meters.
            buffer_size: Number of samples per audio buffer.
            context_len: Number of buffers to retain in the context.
            device: PyTorch device for computation. Defaults to CPU.

        Returns:
            Number of buffers in the reshaped full_rir_rt tensor.
        """
        # Computing early echoes
        early = self.get_early_echoes_at_point(point, source_point, reflection_order=1)
        if early.shape[0] == 0:
            # If there are no early reflections, rescaling is impossible
            self.full_rir_rt = torch.zeros_like(self.late_rir_rt)
            return self.full_rir_rt.shape[0]

        # Blending in with late reverberation
        s = self.estimate_scaling_factor(early, self.late_rir, window_size=0.001)
        early_max = torch.argmax(torch.abs(early)).item()
        start, end = early_max, early.shape[0]
        time = torch.arange(end - start)
        fade_in = 0.5 * (1 - torch.cos(np.pi * time / (end - start)))
        self.full_rir = self.late_rir.clone() * s
        self.full_rir[:start] = early[:start]
        self.full_rir[start:end] = (
            early[start:end] * (1 - fade_in) + fade_in * s * self.late_rir[start:end]
        )

        # Shaping for real-time convolution
        rest = self.full_rir.shape[0] % buffer_size
        n_buffers = self.full_rir.shape[0] // buffer_size

        self.full_rir_rt = (
            self.full_rir[:-rest]
            .view(n_buffers, buffer_size)[:context_len]
            .flip(0)
            .to(device)
        )
        return self.full_rir_rt.shape[0]

    def blend_early_late(self, early: Tensor, late: Tensor) -> tuple[Tensor, Tensor]:
        """Blend early reflections with late reverberation using cosine crossfade.

        Scales late reverberation to match early energy, applies a smooth cosine
        crossfade transition, and returns blended components that sum to the
        original early+late response.

        Args:
            early: Early reflections impulse response.
            late: Late reverberation impulse response.

        Returns:
            Tuple of (blended_early, blended_late) tensors ready for summation.
        """
        if early.shape[0] == 0:
            # If there are no early reflections, return zeros for early and late (no scaling can be estimated)
            return torch.tensor([0.0]), torch.zeros_like(late)

        s = self.estimate_scaling_factor(early, late, window_size=0.001)
        early_max = torch.argmax(torch.abs(early)).item()
        start, end = early_max, early.shape[0]
        time = torch.arange(end - start)
        fade_in = 0.5 * (1 - torch.cos(np.pi * time / (end - start)))
        blended_late = late.clone() * s
        blended_late[:start] = 0
        blended_late[start:end] = blended_late[start:end] * fade_in
        blended_early = early.clone()
        blended_early[start:end] = blended_early[start:end] * (1 - fade_in)
        return blended_early, blended_late

    def get_early_echoes_at_point(
        self,
        point: PointType,
        source_point: PointType | None = None,
        method: str = "ism",
        reflection_order: int = 4,
        verbose=False,
    ) -> Tensor:
        """Compute early reflections at a receiver point using Image Source Method.

        Generates image source positions, applies visibility checks, and computes
        impulse responses. Can optionally provide detailed timing information.

        Args:
            point: 3D receiver position (x, y, z) in meters. Must be inside room.
            source_point: 3D source position. If None, uses room center.
            method: Algorithm to use. Only "ism" currently supported.
            reflection_order: Maximum number of room reflections to compute.
            verbose: If True, prints detailed timing breakdown of computation steps.

        Returns:
            Tensor containing the early reflections impulse response.

        Raises:
            ValueError: If method is not "ism" or if points are outside the room.
        """
        tik = perf_counter()
        if not self.room.is_inside(np.array(point).reshape(1, 3)):
            raise ValueError("The point should be inside the room.")
        match method:
            case "ism":
                # ----------------------------------------------------------------------
                # Initialization of the ISM
                # ----------------------------------------------------------------------

                # Set the microphone position
                if self.room.room.mic_array is None:
                    self.room.room.add_microphone(point)
                else:
                    self.room.room.mic_array.R[:, 0] = point

                # Set the source position
                if source_point is None:
                    bbox = self.room.room.get_bbox()
                    source_point = np.mean(bbox, axis=1)

                if not self.room.is_inside(np.array(source_point).reshape(1, 3)):
                    raise ValueError("The source point is not in the room.")

                if len(self.room.room.sources) == 0:
                    self.room.room.add_source(source_point)
                else:
                    self.room.room.sources[0] = SoundSource(source_point)

                self.room.room.max_order = reflection_order
                self.room.room._init_room_engine()
                self.room.room.room_engine.add_mic(
                    self.room.room.mic_array.R[:, None, 0]
                )

                tok = perf_counter()
                t_init = tok - tik
                # ----------------------------------------------------------------------
                # End - Initialization of the ISM
                # ----------------------------------------------------------------------

                tik = perf_counter()
                self.room.room.image_source_model()
                visibility = cast(list[np.ndarray], self.room.room.visibility)
                tok = perf_counter()
                t_miror = tok - tik

                if np.all(visibility[0] == 0):
                    return torch.tensor([])
                else:
                    tik = perf_counter()
                    ir_ism = compute_ism_rir(
                        self.room.room.sources[0],
                        point,
                        None,
                        visibility[0][0, :],
                        pra_constants.get("frac_delay_length"),
                        self.room.room.c,
                        self.room.room.fs,
                        self.room.room.octave_bands,
                        air_abs_coeffs=self.room.room.air_absorption,
                        min_phase=self.room.room.min_phase,
                    )
                    tok = perf_counter()
                t_rir = tok - tik

                tik = perf_counter()
                torch_ism = torch.from_numpy(ir_ism.copy())
                tok = perf_counter()
                t_conv = tok - tik

                t_tot = t_init + t_miror + t_rir + t_conv

                if not verbose:
                    return torch_ism

                print("------------- Parameters -----------------")
                print("ray_t needed", self.room.room.simulator_state["rt_needed"])
                print("random", self.room.room.simulator_state["random_ism_needed"])
                print("air absorption", self.room.room.air_absorption)
                print("min phase", self.room.room.min_phase)
                print("------------- Parameters -----------------")
                print("Reflection order", self.room.room.max_order)
                print(f"ISM setup took {t_init:.2e} seconds ({100*t_init/t_tot:.1f}%).")
                print(f"ISM took {t_miror:.2e} seconds ({100*t_miror/t_tot:.1f}%).")
                print(f"RIR took {t_rir:.2e} seconds ({100*t_rir/t_tot:.1f}%).")
                print(f"To torch took {t_conv:.2e} seconds ({100*t_conv/t_tot:.1f}%).")
                print(f"Total time: {t_tot:.2e} seconds ({1/ t_tot:.1f} it/s).")

                return torch_ism

            case _:
                raise ValueError(f"Unknown method {method} for early echoes synthesis.")

    def get_modes_at_point(self, point: PointType) -> Tensor:
        """Compute the modal decomposition (frequency response) at a point.

        Extracts the frequency response at the receiver point, performs minimum
        phase reconstruction via cepstral analysis, and returns the modal impulse
        response.

        Args:
            point: 3D receiver position (x, y, z) in meters.

        Returns:
            Tensor containing the modal impulse response at the point.

        Raises:
            ValueError: If point coordinates are not 3D.
        """
        if not len(point) == 3:
            raise ValueError("Point coordinates should be 3D (x, y, z)")
        # Frequency response at this point
        freq_modes_octave = self.room.get_frequency_response_at_point(point)
        linear_freqs = torch.linspace(0, self.center_freqs[-1], 64)
        interp = PchipInterpolator(self.center_freqs, freq_modes_octave)
        freq_modes_linear = torch.from_numpy(interp(linear_freqs))
        cepstrum = cast(Tensor, torch.fft.irfft(torch.log(freq_modes_linear)))
        l = cepstrum.shape[-1]
        cepstrum[1 : l // 2] *= 2
        cepstrum[l // 2 + 1 :] = 0
        modes_at_begin = torch.exp(torch.fft.rfft(cepstrum, 64))
        return torch.fft.irfft(modes_at_begin)

    def get_rir_at_point(
        self,
        point: PointType,
        source_point: PointType | None = None,
        order: int = 3,
        scaling_window_size: float = 0.001,
    ) -> Tensor:
        """Compute complete RIR (early + late) at a receiver point.

        Combines early reflections (via ISM) with late reverberation using the
        modal frequency response and applies a smooth crossfade transition.

        Args:
            point: 3D receiver position (x, y, z) in meters.
            source_point: 3D source position. If None, uses room center.
            order: Reflection order for early reflections computation.
            scaling_window_size: Window duration (seconds) for energy-based scaling.

        Returns:
            Tensor containing the complete impulse response at the point.
        """
        freq_ir = self.get_modes_at_point(point)
        late = convolve(freq_ir, self.late_rir)
        early = self.get_early_echoes_at_point(
            point,
            reflection_order=order,
            source_point=source_point,
        )
        if early.shape[0] == 0:
            return torch.zeros_like(late)

        s = self.estimate_scaling_factor(early, late, window_size=scaling_window_size)
        argmax_early = torch.argmax(torch.abs(early)).item()
        start_fed_in = int(argmax_early)
        end_fed_in = early.shape[0]
        t = torch.arange(end_fed_in - start_fed_in)
        fed_in = 0.5 * (
            1 - torch.cos(np.pi * t / (end_fed_in - start_fed_in))
        )  # Cosine fade-in
        late[start_fed_in:end_fed_in] = late[start_fed_in:end_fed_in] * fed_in
        late[:start_fed_in] = 0
        early[start_fed_in:end_fed_in] = early[start_fed_in:end_fed_in] * (1 - fed_in)
        early[end_fed_in:] = 0
        if early.shape[0] >= late.shape[0]:
            return early
        else:
            late *= s
            late[: early.shape[0]] += early
            return late

    def estimate_scaling_factor(
        self, early: Tensor, late: Tensor, window_size=1e-3
    ) -> float:
        """Estimate energy scaling factor to match early and late reverberation.

        Computes sliding-window RMS energy comparison after the direct sound,
        uses robust median-based estimation, and returns the sqrt of ratio
        to normalize amplitudes.

        Args:
            early: Early reflections impulse response.
            late: Late reverberation impulse response.
            window_size: Analysis window duration in seconds. Defaults to 1 ms.

        Returns:
            Scaling factor (float) to apply to late reverberation.
        """
        # Estimate the scaling factor between early and late reverberation by comparing their energy in short windows after the direct sound
        window_sample_size = int(window_size * self.sr)
        windows_start_samples = torch.arange(
            0,
            min(early.shape[0], late.shape[0]) - window_sample_size,
            window_sample_size,
        )
        win_start_samples_expanded = windows_start_samples.unsqueeze(1) + torch.arange(
            window_sample_size
        )
        early_windows = early[win_start_samples_expanded]
        late_windows = late[win_start_samples_expanded]
        early_energy = torch.sum(early_windows**2, dim=1)
        late_energy = torch.sum(late_windows**2, dim=1)
        discard_mask = early_energy < 1e-3 * torch.max(early_energy)
        # Discard windows with very low energy (around 0.1% of the max energy)
        early_energy = early_energy[~discard_mask]
        late_energy = late_energy[~discard_mask]
        scale_estimates = early_energy / (late_energy + 1e-8)
        # Return the median of the scale estimates to be robust to outliers
        median_scale = torch.sqrt(torch.median(scale_estimates))
        return median_scale.item()

    def apply_reverb_at_point(
        self,
        source: Tensor,
        point: PointType,
        source_point: PointType | None = None,
    ) -> Tensor:
        """Apply reverberation to audio signal for a specific receiver point.

        Computes the RIR at the receiver point and applies it via frequency-domain
        convolution to the input source audio.

        Args:
            source: Input audio signal tensor.
            point: 3D receiver position (x, y, z) in meters.
            source_point: 3D source position. If None, uses room center.

        Returns:
            Tensor containing the reverberant audio signal.
        """
        rir = self.get_rir_at_point(point, source_point=source_point)
        return convolve(source, rir.to(source))
