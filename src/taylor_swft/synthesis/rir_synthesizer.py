from ..utils import constants
from matplotlib.figure import Figure
from torch import Tensor
from torch.nn import Module, Parameter
from typing import cast
from warnings import warn
import torch


class PMatrix(Module):
    """PyTorch module implementing the P operator for late RIR synthesis.

    Applies an exponential decay operator to input signals based on an
    RT60 profile using Taylor series expansion in the frequency domain.
    """

    def __init__(
        self,
        rt_60_profile: Tensor,
        sample_rate: int,
        requires_grad: bool = False,
    ) -> None:
        """Initialize PMatrix with RT60 profile and sample rate.

        Args:
            rt_60_profile: Tensor containing the RT60 reverberation time profile
                in the frequency domain.
            sample_rate: Audio sample rate in Hz.
            requires_grad: Whether to enable gradient computation for the RT60
                profile. Defaults to False.
        """
        super().__init__()
        self.rt_60_profile = Parameter(rt_60_profile)
        self.sample_rate = sample_rate
        self.delta_alpha_bar = constants.ALPHA_BAR_DELTA
        if not requires_grad:
            self.requires_grad_(False)

    def compute_constants(self, x: Tensor) -> None:
        """Compute and cache FFT size and tensor device/dtype from input.

        Args:
            x: Input tensor used to determine sequence length and compute FFT size.
        """
        self.Lh = x.shape[-1]
        self.nfft = int(self.Lh * constants.TAYLOR_NFFT_LEN_RATIO)
        self.tensor_args = {
            "dtype": x.dtype,
            "device": x.device,
        }
        self.to(x.device)

    @property
    def omega(self) -> Tensor:
        """Compute frequency-domain exponential coefficients.

        Returns:
            Tensor of complex exponentials exp(-2j * pi * k / nfft)
            for k in [0, nfft/2+1), used in Fourier analysis.
        """
        return torch.exp(
            -2j
            * torch.pi
            * torch.arange(self.nfft // 2 + 1, **self.tensor_args)
            / self.nfft
        )

    @property
    def taylor_arange(self) -> Tensor:
        """Generate range tensor for Taylor expansion.

        Returns:
            Tensor of integers [0, 1, ..., Lh-1] with shape matching the
            input length, used in Taylor series coefficient computations.
        """
        return torch.arange(self.Lh, **self.tensor_args)

    @property
    def interpolated_rt(self) -> Tensor:
        """Interpolate RT60 profile to FFT bin resolution using cepstral liftering.

        Returns:
            Tensor of interpolated RT60 values with shape (nfft//2+1,).
        """
        cepstrum = cast(Tensor, torch.fft.irfft(torch.log(self.rt_60_profile)))
        cepstrum[1 : cepstrum.shape[-1] // 2] *= 2
        cepstrum[cepstrum.shape[-1] // 2 + 1 :] = 0
        return torch.exp(torch.fft.rfft(cepstrum, self.nfft)).abs()

    @property
    def P_fourier(self) -> Tensor:
        """Compute frequency-domain decay operator from interpolated RT60.

        Returns:
            Tensor of complex exponential decay coefficients derived from
            the interpolated RT60 profile.
        """
        return torch.exp(
            -(3 * torch.log(torch.tensor(10.0, **self.tensor_args)))
            / (self.interpolated_rt * self.sample_rate)
        )

    def taylor_mul(self, x: Tensor, order=500) -> Tensor:
        """Apply P operator using Taylor series expansion.

        Implements efficient frequency-domain convolution with exponential decay
        operator using Taylor series approximation up to specified order.

        Args:
            x: Input signal tensor of shape (..., Lh).
            order: Maximum Taylor series order. Defaults to 500.

        Returns:
            Output tensor after applying P operator, same shape as input.

        Warns:
            UserWarning: If Taylor series diverges before reaching specified
                order due to numerical instability (inf or nan values).
        """
        self.compute_constants(x)
        alpha_bar = -torch.log(
            (self.P_fourier.abs().max() + self.P_fourier.abs().min()) / 2
            - self.delta_alpha_bar
        )
        e = torch.exp(-self.taylor_arange * alpha_bar.to(x.device))
        x = x * e
        Y_fourier = cast(Tensor, torch.fft.rfft(x, self.nfft))
        diff = (torch.exp(alpha_bar) * self.P_fourier).abs() - 1
        diff = diff * self.omega
        diff_power = torch.ones_like(diff)
        diff_max = diff.abs().max()
        diff = diff / diff_max

        for k in range(1, min(order, self.Lh)):
            x = (x * self.taylor_arange[: self.Lh - k + 1])[..., 1:] / k * diff_max
            diff_power = diff_power * diff
            next_term = cast(Tensor, torch.fft.rfft(x, self.nfft) * diff_power)
            if next_term.abs().max() < constants.ALMOST_ZERO:
                break
            has_inf = torch.isinf(next_term).any()
            has_nan = torch.isnan(next_term).any()
            if has_inf or has_nan:
                warn(
                    f"Taylor series diverged at order {k}. Stopping iteration. "
                    f"Max abs of next term: {next_term.abs().max().item()}. "
                    f"Has inf: {has_inf}. Has NaN: {has_nan}."
                )
                break
            Y_fourier = Y_fourier + next_term
        y = cast(Tensor, torch.fft.irfft(Y_fourier, self.nfft))[..., : self.Lh]
        return y.real

    def transpose_mul(self, x: Tensor) -> Tensor:
        """Apply transpose of P operator to input signal.

        Naive implementation computing P^T by direct convolution for each
        frequency bin. Very slow for long inputs.

        Args:
            x: Input signal tensor of shape (..., Lh).

        Returns:
            Output tensor after applying P^T operator, same shape as input.

        Warns:
            UserWarning: Always issued, as this is a naive unoptimized
                implementation suitable only for short signals.
        """
        warn(
            "This is a naive implementation of the transpose multiplication, it is not optimized for speed. The operation may be very slow for long inputs."
        )
        self.compute_constants(x)
        X_fourier = torch.fft.rfft(x, self.nfft)
        P_fourier = self.P_fourier.abs()
        P_fourier = P_fourier.unsqueeze(0).expand_as(X_fourier)
        y = torch.zeros_like(x, **self.tensor_args)
        for k in range(0, y.shape[-1]):
            y[..., k] = torch.fft.irfft(X_fourier * P_fourier**k)[..., k]
        return y[..., : self.Lh].real

    def taylor_mul_impulse(self, impulse: Tensor, n: int) -> Tensor:
        """Apply P operator to impulse signal with one non-zero entry.

        Args:
            impulse: Input impulse tensor with single non-zero entry.
            n: Index of the impulse.

        Returns:
            Output tensor after applying P operator.

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        raise NotImplementedError("taylor_mul_impulse is not implemented yet.")

    def plot_spectrum(self, fig: Figure) -> None:
        """Plot spectrogram of late RIR with RT profile overlay.

        Args:
            fig: Matplotlib figure object to plot on.
        """
        freqs = torch.linspace(0, self.sample_rate // 2, self.nfft // 2 + 1)
        rt = self.interpolated_rt

        eps = torch.randn(int(1.2 * rt.max() * self.sample_rate))
        late_rir = self.taylor_mul(eps)

        ax = fig.gca()
        _, _, _, im = ax.specgram(late_rir, Fs=self.sample_rate, vmin=-100)
        fig.colorbar(im, ax=ax, label="Magnitude (dB)")
        ax.plot(rt, freqs, "r", label="RT profile")
        ax.set_title("Late RIR Spectrogram with RT Profile")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.legend()
