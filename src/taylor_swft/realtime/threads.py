from ..room.graphics import GraphicsContextManager
from ..utils.custom_typing import PointType
from .real_time import SWFTContext
from itertools import cycle
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np
import queue
import threading
import torch
from numpy.typing import NDArray


class SWFTThread:
    """Thread for real-time SWFT audio processing.

    Runs audio processing buffers through the SWFT reverberator in a separate
    thread synchronized by events. Supports both interactive and trajectory-based
    position updates.

    Attributes:
        swft_context (SWFTContext): Audio processing context.
        inner_thread (threading.Thread): Worker thread instance.
        store_output (bool): Whether to accumulate output buffers.
        get_positions_method (str): "interactive", "trajectories", or "undefined".
    """

    def __init__(
        self,
        swft_context: SWFTContext,
        input_audio: Tensor,
        synchro_flag: threading.Event,
        stop_all_flag: threading.Event,
        store_output: bool = False,
        loop_input_audio: bool = False,
    ):
        """Initialize SWFT processing thread.

        Args:
            swft_context: Configured SWFTContext for audio processing.
            input_audio: Input audio tensor of shape (samples, channels).
            synchro_flag: Event for thread synchronization.
            stop_all_flag: Event to signal thread termination.
            store_output: If True, accumulate all output buffers. Defaults to False.
            loop_input_audio: If True, loop input when exhausted. Defaults to False.
        """
        super().__init__()
        self.swft_context = swft_context
        self.inner_thread = threading.Thread(target=self.run)
        self.synchro_flag = synchro_flag
        self.stop_event = stop_all_flag
        self.store_output = store_output
        self.full_output: list[Tensor] = []
        self.get_positions_method = "undefined"
        self.buffer_dur = swft_context.buffer_size / swft_context.rev.sr
        self.loop_input_audio = loop_input_audio
        truncated_input_len = (
            input_audio.shape[0] // swft_context.buffer_size
        ) * swft_context.buffer_size
        self.input_audio_buffers = input_audio[:truncated_input_len].reshape(
            -1, swft_context.buffer_size, swft_context.n_channels
        )

        # to communicate exceptions from the thread to the main thread, since exceptions in threads don't propagate to the main thread by default
        self.error_bin = queue.Queue()

        self.input_buffer_generator = self.get_input_buffer_generator()

    def get_input_buffer_generator(self):
        """Create input buffer generator (cyclic or one-pass).

        Returns:
            Iterator or cycle generator over input audio buffers.
        """
        if self.loop_input_audio:
            return cycle(self.input_audio_buffers)
        else:
            return iter(self.input_audio_buffers)

    def run(self):
        while not self.stop_event.is_set():
            try:
                mic_pos, source_pos = self.get_current_mic_and_source_positions()

                input_buffer = self.get_next_input_buffer()
                out_buffer = self.swft_context.process_next_buffer(
                    input_buffer, mic_pos, source_pos
                )
                if self.store_output:
                    self.full_output.append(out_buffer)
            except Exception as e:
                self.error_bin.put(e)
                self.stop_event.set()
                break
            self.synchro_flag.wait()

    def set_interactive_mode(self, graphics_context: GraphicsContextManager):
        """Enable interactive mode with mouse-controlled positions.

        Args:
            graphics_context: GraphicsContextManager for position queries.

        Raises:
            AssertionError: If position method already set.
        """
        assert self.get_positions_method == "undefined", (
            "Position method already set. Cannot set interactive after trajectories mode. "
            "Please use reset_all() to reset the context if you want to change the get_position method."
        )

        self.get_positions_method = "interactive"
        self.graphics_context = graphics_context

    def set_trajectories_mode(
        self, dynamic_mic_pos: np.typing.NDArray, dynamic_source_pos: np.typing.NDArray
    ):
        """Enable trajectory mode with predefined position sequences.

        Args:
            dynamic_mic_pos: Array of shape (n_positions, 3) with mic positions.
            dynamic_source_pos: Array of shape (n_positions, 3) with source positions.

        Raises:
            AssertionError: If position method already set or arrays invalid shape.
        """
        assert self.get_positions_method == "undefined", (
            "Position method already set. Cannot set trajectories mode after interactive mode. "
            "Please use reset_all() to reset the context if you want to change the get_position method."
        )

        self.get_positions_method = "trajectories"
        assert (
            len(dynamic_mic_pos.shape) == 2
        ), f"dynamic_mic_pos should be of shape (n_positions, 3) but is {len(dynamic_mic_pos.shape)} dimensional"
        assert (
            len(dynamic_source_pos.shape) == 2
        ), f"dynamic_source_pos should be of shape (n_positions, 3) but is {len(dynamic_source_pos.shape)} dimensional"
        assert (
            dynamic_mic_pos.shape[1] == 3
        ), f"dynamic_mic_pos should have shape (n_positions, 3) but has shape {dynamic_mic_pos.shape}"
        assert (
            dynamic_source_pos.shape[1] == 3
        ), f"dynamic_source_pos should have shape (n_positions, 3) but has shape {dynamic_source_pos.shape}"

        self.mic_pos_generator = cycle(dynamic_mic_pos)
        self.source_pos_generator = cycle(dynamic_source_pos)

    def reset_all(self):
        """Reset thread state for fresh processing session.

        Stops thread if running, clears position generators, flags, and state.
        Position mode must be set again before starting.
        """
        if self.inner_thread.is_alive():
            self.stop()
            self.join()
        self.get_positions_method = "undefined"
        if hasattr(self, "graphics_context"):
            del self.graphics_context
        if hasattr(self, "mic_pos_generator"):
            del self.mic_pos_generator
        if hasattr(self, "source_pos_generator"):
            del self.source_pos_generator
        del self.inner_thread
        self.inner_thread = threading.Thread(target=self.run)
        self.stop_event.clear()
        self.swft_context.reset()
        self.full_output: list[Tensor] = []
        self.error_bin = queue.Queue()
        self.input_buffer_generator = self.get_input_buffer_generator()

    def get_current_mic_and_source_positions(self) -> tuple[NDArray, NDArray]:
        """Get next mic and source positions (interactive or trajectory).

        Returns:
            Tuple of (mic_pos, source_pos) as PointType (3D coordinates).

        Raises:
            RuntimeError: If position method not set.
        """
        mic_pos: NDArray
        source_pos: NDArray

        if self.get_positions_method == "interactive":
            mic_pos = self.graphics_context.microphone_coordinates.copy()
            source_pos = self.graphics_context.source_coordinates.copy()
        elif self.get_positions_method == "trajectories":
            mic_pos = next(self.mic_pos_generator)
            source_pos = next(self.source_pos_generator)
        else:
            raise RuntimeError(
                "No method set to get current mic and source positions. "
                "Please call set_trajectories_mode() or set_interactive_mode() before trying to get positions."
            )

        return mic_pos, source_pos

    def get_next_output_buffer(self):
        """Retrieve next processed output buffer from context.

        Returns:
            Tensor of processed audio output.
        """
        return self.swft_context.get_next_output_buffer()

    def get_next_input_buffer(self):
        """Get next input buffer from generator.

        Returns:
            Tensor of next audio input buffer.

        Raises:
            StopIteration: If input exhausted and loop_input_audio=False.
        """
        return next(self.input_buffer_generator)

    def get_full_output(self):
        """Retrieve concatenated output from all processed buffers.

        Returns:
            Tensor of all accumulated output (concatenated along sample axis).

        Raises:
            ValueError: If store_output was False.
        """
        if not self.store_output:
            raise ValueError("store_output is False, full output not available.")
        return torch.cat(self.full_output, dim=0)

    def start(self):
        """Start the processing thread.

        Raises:
            RuntimeError: If no position method set.
        """
        if self.get_positions_method == "undefined":
            raise RuntimeError(
                "No method set to get current mic and source positions. "
                "Please call set_trajectories_mode() or set_interactive_mode() before starting the thread."
            )
        self.inner_thread.start()

    def stop(self):
        self.stop_event.set()

    def join(self):
        """Wait for thread to finish execution."""
        self.inner_thread.join()


class GraphicsThread:
    """Thread for rendering graphics updates synchronized with audio processing.

    Updates graphics context at fixed intervals coordinated with audio buffers.
    Supports both interactive and trajectory-based visualization.

    Attributes:
        graphics_context (GraphicsContextManager): Rendering context.
        interactive_mode (bool): True if mouse-controlled, False if trajectory.
        mic_trajectory_memory (list): Recorded microphone positions.
        source_trajectory_memory (list): Recorded source positions.
    """

    def __init__(
        self,
        graphics_context: GraphicsContextManager,
        synchro_flag: threading.Event,
        stop_all_flag: threading.Event,
        dynamic_mic_pos: Tensor = torch.tensor([]),
        dynamic_source_pos: Tensor = torch.tensor([]),
    ):
        """Initialize graphics rendering thread.

        Args:
            graphics_context: GraphicsContextManager for rendering.
            synchro_flag: Event for thread synchronization.
            stop_all_flag: Event to signal termination.
            dynamic_mic_pos: Optional trajectory of shape (n_positions, 3).
                If provided with dynamic_source_pos, enables trajectory mode. Defaults to empty.
            dynamic_source_pos: Optional trajectory of shape (n_positions, 3).
                If provided with dynamic_mic_pos, enables trajectory mode. Defaults to empty.

        Raises:
            AssertionError: If graphics context mode mismatches position data.
        """
        self.graphics_context = graphics_context
        if len(dynamic_mic_pos) > 0 and len(dynamic_source_pos) > 0:
            assert (
                self.graphics_context.disable_interactive == True
            ), "Graphics context must be initialized with disable_interactive=True when providing dynamic positions."
            self.mic_pos_generator = cycle(dynamic_mic_pos)
            self.source_pos_generator = cycle(dynamic_source_pos)
            self.interactive_mode = False
        elif len(dynamic_mic_pos) == 0 and len(dynamic_source_pos) == 0:
            assert (
                self.graphics_context.disable_interactive == False
            ), "Graphics context must be initialized with disable_interactive=False for interactive mode."
            self.interactive_mode = True
        self.stop_event = stop_all_flag

        self.mic_trajectory_memory: list[np.typing.NDArray] = []
        self.source_trajectory_memory: list[np.typing.NDArray] = []
        self.synchro_flag = synchro_flag
        self.inner_thread = threading.Thread(target=self.run)
        self.error_bin = queue.Queue()

    def start(self):
        """Start graphics rendering thread and initialize graphics context."""
        self.graphics_context.__enter__()
        self.inner_thread.start()

    def run(self):
        """Main rendering loop synchronized with audio processing.

        Updates graphics, records trajectories, and waits for sync signal.
        """
        while not self.stop_event.is_set():
            try:
                if not self.interactive_mode:
                    mic_pos = next(self.mic_pos_generator).numpy()
                    source_pos = next(self.source_pos_generator).numpy()
                    self.graphics_context.microphone_coordinates = mic_pos
                    self.graphics_context.source_coordinates = source_pos
                    self.graphics_context.draw_source_and_mic()
                self.mic_trajectory_memory.append(
                    self.graphics_context.microphone_coordinates.copy()
                )
                self.source_trajectory_memory.append(
                    self.graphics_context.source_coordinates.copy()
                )
            except Exception as e:
                self.error_bin.put(e)
                self.stop_event.set()
                break
            self.synchro_flag.wait()

    def stop(self):
        self.stop_event.set()
        self.graphics_context.__exit__(None, None, None)

    def join(self):
        """Wait for graphics thread to finish."""
        self.inner_thread.join()

    def show(self):
        """Display the graphics window."""
        self.graphics_context.show()

    def monitor_show(self):
        """Continuously display graphics while thread is running.

        Shows graphics updates synchronized with audio buffer processing.
        """
        while self.inner_thread.is_alive() and not self.stop_event.is_set():
            self.synchro_flag.wait()
            self.graphics_context.show()

    def plot_trajectories(self):
        """Visualize recorded mic and source trajectories after processing.

        Creates matplotlib plot with trajectory arrows and paths overlaid on
        the room layout. Requires at least one iteration of position updates.

        Raises:
            RuntimeError: If no trajectory data recorded (thread not run or not updated).
        """
        if (
            len(self.mic_trajectory_memory) == 0
            or len(self.source_trajectory_memory) == 0
        ):
            raise RuntimeError(
                "No trajectory data available to plot. Make sure to run the graphics thread and update the graphics context before trying to plot trajectories."
            )
        img = self.graphics_context.img.copy()
        # remove the escape key text in the image by adding a white rectangle over it
        img[10:50, 10:500] = 255
        mic_positions = []
        source_positions = []
        for mic_pos, source_pos in zip(
            self.mic_trajectory_memory, self.source_trajectory_memory
        ):
            mic_pos_pix = self.graphics_context._coordinates_to_pixels(*mic_pos[:2])
            source_pos_pix = self.graphics_context._coordinates_to_pixels(
                *source_pos[:2]
            )
            mic_positions.append(mic_pos_pix)
            source_positions.append(source_pos_pix)
        source_positions = np.array(source_positions)
        mic_positions = np.array(mic_positions)
        arrow_idx = len(source_positions) // 3
        traj_mic_color = np.array(self.graphics_context.mic_color) / 255.0 + 0.2
        traj_source_color = np.array(self.graphics_context.source_color) / 255.0 + 0.2
        # clamp colors to [0, 1]
        traj_mic_color = np.clip(traj_mic_color, 0, 1)
        traj_source_color = np.clip(traj_source_color, 0, 1)
        plt.imshow(img)
        plt.title("Microphone and Source Trajectories")
        plt.arrow(
            mic_positions[arrow_idx, 0],
            mic_positions[arrow_idx, 1],
            mic_positions[arrow_idx + 5, 0] - mic_positions[arrow_idx, 0],
            mic_positions[arrow_idx + 5, 1] - mic_positions[arrow_idx, 1],
            head_width=20,
            head_length=20,
            fc=traj_mic_color,
            ec=traj_mic_color,
        )
        plt.arrow(
            source_positions[arrow_idx, 0],
            source_positions[arrow_idx, 1],
            source_positions[arrow_idx + 5, 0] - source_positions[arrow_idx, 0],
            source_positions[arrow_idx + 5, 1] - source_positions[arrow_idx, 1],
            head_width=20,
            head_length=20,
            fc=traj_source_color,
            ec=traj_source_color,
        )
        plt.plot(
            source_positions[:, 0],
            source_positions[:, 1],
            color=traj_source_color,
        )
        plt.plot(
            mic_positions[:, 0],
            mic_positions[:, 1],
            color=traj_mic_color,
        )
        plt.legend()
