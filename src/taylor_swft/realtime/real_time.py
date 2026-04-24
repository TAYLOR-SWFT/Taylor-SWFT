from ..core.taylor_swft import Reverberator
from ..utils.custom_typing import PointType
from torch import Tensor
from torchaudio.functional import fftconvolve
import torch
from numpy.typing import NDArray


def convolve_one_buffer(source: Tensor, ir: Tensor, buffer_size: int) -> Tensor:
    """Convolve source signal with impulse response, returning valid output.

    Performs FFT-based convolution and returns the last buffer_size samples
    of the valid convolution (excluding the initial transient).

    Args:
        source: Input signal tensor.
        ir: Impulse response tensor.
        buffer_size: Size of output buffer to return.

    Returns:
        Tensor: Convolved signal of length buffer_size or greater.
    """
    ir = ir.to(source)
    res = fftconvolve(
        source[len(source) - len(ir) - buffer_size + 1 :], ir, mode="valid"
    )
    return res


class CircularBuffer:
    """Circular buffer for managing audio chunks.

    Stores multiple audio chunks in a ring buffer and supports windowed
    overlap-add processing for smooth transitions between chunks.

    Attributes:
        buffer (Tensor): Ring buffer of shape (context_len, buffer_size, n_channels).
        ptr (int): Current write position in the ring buffer.
        size (int): Number of chunks the buffer can store (context_len).
        buffer_size (int): Size of each audio chunk in samples.
        tail_buffer (Tensor): Buffer for storing tail data if needed.
        overlap (bool): Whether to use overlap-add windowing.
        window (Tensor): Hann window for overlap-add if enabled.
    """

    def __init__(
        self,
        context_len: int,
        buffer_size: int,
        n_channels: int,
        device="cpu",
        overlap=False,
    ) -> None:
        """Initialize the circular buffer.

        Args:
            context_len: Number of buffer chunks to store (context length).
            buffer_size: Samples per chunk.
            n_channels: Number of audio channels.
            device: PyTorch device (e.g., "cpu" or "cuda"). Defaults to "cpu".
            overlap: If True, use Hann window for overlap-add. Defaults to False.
        """
        self.buffer = torch.zeros((context_len, buffer_size, n_channels), device=device)
        self.ptr = 0
        self.size = context_len
        self.buffer_size = buffer_size
        self.tail_buffer = torch.zeros((buffer_size, n_channels), device=device)
        self.overlap = overlap
        if overlap:
            self.window = torch.hann_window(buffer_size * 2, device=device).unsqueeze(
                -1
            )

    def append(self, chunk: Tensor) -> None:
        """Append a new audio chunk to the circular buffer.

        Args:
            chunk: Audio tensor of shape (chunk_size, n_channels).
                If overlap is True, chunk_size must be 2*buffer_size.

        Raises:
            AssertionError: If overlap is enabled and chunk size doesn't match.
        """
        next_ptr = (self.ptr + 1) % self.size
        if self.overlap:
            assert (
                chunk.shape[0] == self.window.shape[0]
            ), "Chunk size must be 2*buffer_size when using overlap-add with a Hann window."
            win_chunk = chunk * self.window
            self.buffer[self.ptr] += win_chunk[: self.buffer_size]
            self.buffer[next_ptr].copy_(win_chunk[self.buffer_size :])
        else:
            self.buffer[self.ptr].copy_(chunk)
        self.ptr = next_ptr

    def get(self) -> Tensor:
        """Return buffer contents in chronological order.

        Returns:
            Tensor: Buffer of shape (context_len, buffer_size, n_channels) with
                oldest data first.
        """
        if self.ptr == 0:
            return self.buffer
        else:
            return torch.cat((self.buffer[self.ptr :], self.buffer[: self.ptr]), dim=0)


class SWFTContext:
    """Context class for real-time SWFT processing.
    Attributes:
        rev (Reverberator): Reverberator instance handling the core computations.
        n_channels (int): Number of audio channels (only 1 is supported for now).
        buffer_size (int): Size of audio buffers in samples.
        input_circular_buffer (CircularBuffer): Buffer for storing recent input chunks.
        filtered_input_buffer (CircularBuffer): Buffer for storing recent filtered input chunks.
        late_coloration (Tensor): Late reverberation impulse response for coloration (result of the product P.dot(noise)).
        reflection_order (int): Reflection order for early echoes.
        device: PyTorch device for processing.
        window: Hann window for overlap-add if enabled.
        previous_output_chunk: Buffer for storing tail of previous output when using overlap-add.
        overlap: Whether to use overlap-add processing.
        output_queue: Queue for storing processed output buffers ready for retrieval.
    """

    def __init__(
        self,
        rev: Reverberator,
        buffer_dur: float = 30e-3,
        n_channels: int = 1,
        reflection_order: int = 2,
        overlap: bool = True,
        device="cpu",
    ):
        """
        Initialize the SWFTContext.
        Args:
            rev: Reverberator instance handling the core computations.
            buffer_dur: Duration of audio buffers in seconds (default: 30e-3).
            n_channels: Number of audio channels (default: 1, higher values are not supported yet).
            reflection_order: Reflection order for early echoes (default: 2). Higher values will increase computational load but may improve early echo quality.
            overlap: Whether to use overlap-add processing (default: True) between output buffers.
            device: PyTorch device for processing (default: "cpu").
        """
        self.rev = rev
        self.n_channels = n_channels

        rir_dur = rev.late_rir.shape[-1] / rev.sr
        self.buffer_size = int(buffer_dur * rev.sr)
        context_len = round(rir_dur / buffer_dur) + 3
        self.input_circular_buffer = CircularBuffer(
            context_len, self.buffer_size, n_channels, device, overlap=False
        )
        self.filtered_input_buffer = CircularBuffer(
            context_len, self.buffer_size, n_channels, device, overlap=False
        )
        self.late_coloration = rev.late_rir.clone()
        self.reflection_order = reflection_order
        self.device = device
        self.window = torch.hann_window(2 * self.buffer_size, device=device)
        self.previous_output_chunk = torch.zeros((self.buffer_size), device=device)
        self.overlap = overlap
        self.output_queue = []

    def process_next_buffer(
        self, input_buffer: Tensor, mic_pos: NDArray, source_pos: NDArray
    ) -> Tensor:
        """
        Process the next audio buffer.
        Args:
            input_buffer: Input audio buffer to be processed (shape: [buffer_size, n_channels]).
            mic_pos: Position of the microphone.
            source_pos: Position of the sound source (shape: [3]).
        Returns:
            Processed audio buffer.
        """

        assert (
            input_buffer.shape[0] == self.buffer_size
        ), f"Input buffer size must be equal to {self.buffer_size} samples."
        assert (
            input_buffer.shape[1] == self.n_channels
        ), f"Input buffer must have {self.n_channels} channels."

        assert (
            len(mic_pos) == 3 and len(source_pos) == 3
        ), "Microphone and source positions must be 3D coordinates."

        self.input_circular_buffer.append(input_buffer)

        modes_ir = self.rev.get_modes_at_point(mic_pos)

        if torch.norm(torch.tensor(source_pos - mic_pos)) < 0.5:
            # If the source is very close to the microphone, we move it slightly to avoid numerical issues in the ISM computation of early echoes
            torch.manual_seed(
                0
            )  # this ensures that the random direction is the same between successive buffers
            random_direction = torch.randn(3)
            random_direction /= torch.norm(random_direction)
            source_pos = mic_pos + random_direction.numpy() * 0.5
            while not self.rev.room.is_inside(source_pos):
                random_direction = torch.randn(3)
                random_direction /= torch.norm(random_direction)
                source_pos = mic_pos + random_direction.numpy() * 0.5

        early_echoes_ir = self.rev.get_early_echoes_at_point(
            mic_pos,
            source_pos,
            method="ism",
            reflection_order=self.reflection_order,
        )

        # clamp the early echoes to avoid extreme values whene the source is very close to the microphone
        early_echoes_ir = torch.clamp(early_echoes_ir, -0.3, 0.3)

        # Apply the cross_fade between the early and late parts
        early_echoes_ir, late_coloration = self.rev.blend_early_late(
            early_echoes_ir, self.late_coloration
        )

        # Filter the current input buffer with the modes
        input_chunk = self.input_circular_buffer.get().squeeze().flatten()
        filtered_buffer = convolve_one_buffer(
            input_chunk, modes_ir, self.buffer_size
        ).unsqueeze(-1)
        self.filtered_input_buffer.append(filtered_buffer)

        filtered_chunk = self.filtered_input_buffer.get().squeeze().flatten()
        if self.overlap:
            # Convolve the filtered input with the late coloration
            out_late = convolve_one_buffer(
                filtered_chunk, late_coloration, 2 * self.buffer_size
            )

            # Convolve the unfiltered input with the early echoes
            out_early = convolve_one_buffer(
                input_chunk, early_echoes_ir, 2 * self.buffer_size
            )
            # Sum early and late parts
            output_buffer = (out_early + out_late) * self.window
            output_buffer[: self.buffer_size] += self.previous_output_chunk
            self.previous_output_chunk = output_buffer[self.buffer_size :]
        else:
            # Convolve the filtered input with the late coloration
            out_late = convolve_one_buffer(
                filtered_chunk, late_coloration, self.buffer_size
            )
            # Convolve the unfiltered input with the early echoes
            out_early = convolve_one_buffer(
                input_chunk, early_echoes_ir, self.buffer_size
            )
            output_buffer = out_early + out_late

        self.output_queue.append(
            output_buffer[: self.buffer_size].unsqueeze(-1).cpu().numpy()
        )
        return output_buffer[: self.buffer_size]

    def get_next_output_buffer(self) -> Tensor:
        """
        Retrieve the next processed audio buffer from the output queue.
        Returns:
            Tensor: Next processed audio buffer.
        """
        if len(self.output_queue) == 0:
            raise RuntimeError(
                "Output queue is empty. No processed audio available yet."
            )
        else:
            return self.output_queue.pop(0)

    def reset(self):
        """Reset the context and clear all buffers and queues."""

        self.input_circular_buffer = CircularBuffer(
            self.input_circular_buffer.size,
            self.buffer_size,
            self.n_channels,
            self.device,
            overlap=False,
        )
        self.filtered_input_buffer = CircularBuffer(
            self.filtered_input_buffer.size,
            self.buffer_size,
            self.n_channels,
            self.device,
            overlap=False,
        )
        self.previous_output_chunk = torch.zeros((self.buffer_size), device=self.device)
        self.output_queue = []
