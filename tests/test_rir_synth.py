from taylor_swft.room.spatial_model import make_demo_room
from taylor_swft.synthesis.rir_synthesizer import PMatrix
from torch import Tensor
from torchaudio.functional import fftconvolve
from typing import cast
import numpy as np
import torch


def test_imports():
    import taylor_swft.synthesis.rir_synthesizer


def taylor_mul_achilles_ref(p_vec: Tensor, x: Tensor) -> Tensor:
    import numpy as np
    import scipy

    class P_class:
        def __init__(self, p: np.ndarray):
            self.p = p
            self.L_p = self.p.shape[0]

        def dot(self, x: np.ndarray, order=500):
            T = x.shape[0]
            x = np.array(x, dtype=complex)
            L = int(1.1 * T)
            Pk = scipy.fft.fft(self.p, L)
            ek = np.exp(-2j * np.pi * np.arange(L) / L)
            zk = Pk * ek
            zk = np.array(zk, dtype=complex)

            Yk = scipy.fft.fft(x, L)

            diff = zk - ek
            diff_max = np.abs(diff).max()
            diff = diff / diff_max
            x_tilde = x
            diff_pow_i = np.ones(L)
            for i in range(1, order + 1):
                # x_tilde(k) = x(k+i)*C_{k+i}^i
                x_tilde = (x_tilde * np.arange(T - i + 1))[1:] / i * diff_max
                # diff_pow_i = (zk-ek)^i
                diff_pow_i = diff_pow_i * diff
                # Yk = \sum_i (zk-ek)^i * FFT(x_tilde)
                Yk = Yk + diff_pow_i * scipy.fft.fft(x_tilde, L)

            return np.real(scipy.fft.ifft(Yk))[:T]

    p = P_class(p_vec.detach().cpu().numpy())
    y = p.dot(x.detach().cpu().numpy())

    return torch.from_numpy(y).to(x.device).to(x.dtype)


def make_parameters():
    rt_len = 16
    rt_min = 0.01
    high_var = 1e-1
    small_var = 1e-2
    high_var_profile = torch.rand(rt_len) * high_var + rt_min
    small_var_profile = torch.rand(rt_len) * small_var + rt_min
    return high_var_profile, small_var_profile


def test_taylor_mul():
    torch.manual_seed(0)
    sr = 16000
    Lh = 500

    p = torch.tensor(
        [
            1,
            -0.00024094,
            0.00009173,
            -0.00003573,
            0.00005369,
            0.00001789,
            0.0000115,
            -0.00000338,
            0.00000897,
            0.00001151,
        ],
        dtype=torch.float64,
    ) * torch.exp(torch.ones(1) * 6.5e-2)
    x = torch.randn(Lh, dtype=p.dtype) / 3

    nfft = 2**9
    p_fourier = cast(Tensor, torch.fft.rfft(p, nfft))
    rt_profile = 3 * torch.log(torch.tensor(10.0)) / (torch.log(p_fourier.abs()) * sr)

    p_inv = cast(Tensor, torch.fft.irfft(1 / p_fourier, nfft))
    y_ref = taylor_mul_achilles_ref(p_inv, x)

    p_matrix = PMatrix(rt_60_profile=rt_profile, sample_rate=sr)
    y = p_matrix.taylor_mul(x).detach().cpu()

    assert torch.allclose(
        y, y_ref, atol=1e-3
    ), "Taylor multiplication does not match reference implementation."


def test_P_vs_P_transpose():
    from taylor_swft import Reverberator
    from warnings import filterwarnings, resetwarnings

    room = make_demo_room()
    barycenter = np.mean(room.room.get_bbox(), axis=1)
    rev = Reverberator(room)
    g_ir = rev.get_modes_at_point(barycenter).unsqueeze(0)

    N_STATS = 10
    Lh = rev.late_rir.shape[-1] // 10
    M = g_ir.shape[-1]
    noise = torch.randn(N_STATS, Lh)
    p_matrix = rev.late_rir_operator

    P_noise = p_matrix.taylor_mul(noise)

    filterwarnings("ignore", category=UserWarning)
    # Silently ignores the warning about inefficient
    Pt_noise = p_matrix.transpose_mul(noise)
    resetwarnings()

    rirs_P = fftconvolve(P_noise.to(g_ir), g_ir)[..., : -M + 1]
    rirs_Pt = fftconvolve(Pt_noise.to(g_ir), g_ir)[..., : -M + 1]

    cov_P = torch.cov(rirs_P.T)
    cov_Pt = torch.cov(rirs_Pt.T)

    absolute_error = torch.abs(cov_P - cov_Pt).max()
    assert 20 * torch.log10(absolute_error) < -25


def test_device_taylor_mul():
    if not torch.cuda.is_available():
        return
    sr = 16000
    Lh = 10000
    h_rt, _ = make_parameters()
    pm = PMatrix(rt_60_profile=h_rt, sample_rate=sr)

    for device in ["cpu", "cuda", "cpu", "cuda"]:
        x = torch.randn(Lh, device=device)
        y = pm.taylor_mul(x)
        assert (
            y.device == x.device
        ), f"Output device {y.device} does not match input device {x.device}."


def test_dtype_taylor_mul():
    sr = 16000
    Lh = 10000
    h_rt, _ = make_parameters()
    pm = PMatrix(rt_60_profile=h_rt, sample_rate=sr)

    for dtype in [torch.float32, torch.float64, torch.float32, torch.float64]:
        x = torch.randn(Lh, dtype=dtype)
        y = pm.taylor_mul(x)
        assert (
            y.dtype == x.dtype
        ), f"Output dtype {y.dtype} does not match input dtype {x.dtype}."
