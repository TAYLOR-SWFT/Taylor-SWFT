from ..core import Reverberator
from ..room.graphics import GraphicsContextManager
from .real_time import SWFTContext
from .threads import GraphicsThread, SWFTThread
from time import sleep
from torch import Tensor
from warnings import warn
import numpy as np
import queue
import threading
import torch


class TaylorSWFTRealTimeProcessor:
    try:
        from sounddevice import OutputStream

        graphics_thread: GraphicsThread
        output_stream: OutputStream
    except (ImportError, OSError):
        warn("Sounddevice library not found.")

    def __init__(
        self,
        reverb: Reverberator,
        input_audio: Tensor,
        audio_buffer_dur: float = 30e-3,
        n_channels: int = 1,
        reflection_order: int = 2,
        overlap: bool = True,
        device="cpu",
        store_full_output: bool = False,
        loop_input_audio: bool = False,
    ):
        """Initialize real-time SWFT reverb processor with threading support.

        Sets up audio and position tracking threads that run asynchronously.
        Audio processing and graphics can be synchronized via threading events.

        Args:
            reverb: Initialized Reverberator instance with room model.
            input_audio: Input audio tensor to process (samples, channels).
            audio_buffer_dur: Duration of each audio buffer in seconds. Defaults to 30 ms.
            n_channels: Number of audio channels. Defaults to 1.
            reflection_order: ISM reflection order for early echoes. Defaults to 2.
            overlap: If True, use overlap-add windowing. Defaults to True.
            device: PyTorch device ("cpu" or "cuda"). Defaults to "cpu".
            store_full_output: If True, accumulate all output for retrieval. Defaults to False.
            loop_input_audio: If True, loop input audio when exhausted. Defaults to False.
        """
        self.reverb = reverb
        self.audio_sr = reverb.sr
        self.n_channels = n_channels
        self.audio_buffer_size = int(audio_buffer_dur * self.audio_sr)
        self.audio_buffer_dur = self.audio_buffer_size / self.audio_sr
        self.synchro_flag = threading.Event()
        self.stop_all_flag = threading.Event()
        self.swft_context = SWFTContext(
            rev=reverb,
            buffer_dur=audio_buffer_dur,
            n_channels=n_channels,
            reflection_order=reflection_order,
            overlap=overlap,
            device=device,
        )
        self.swft_thread = SWFTThread(
            self.swft_context,
            input_audio=input_audio,
            store_output=store_full_output,
            synchro_flag=self.synchro_flag,
            stop_all_flag=self.stop_all_flag,
            loop_input_audio=loop_input_audio,
        )
        self.conductor = threading.Thread(target=self.run_conductor)
        self.is_mode_set = False

    def plot_trajectories(self):
        """Plot microphone and source trajectories after processing.

        Visualizes recorded position trajectories from the graphics thread.
        Only available if graphics_thread was initialized.

        Raises:
            RuntimeError: If graphics_thread not initialized.
        """
        if hasattr(self, "graphics_thread"):
            self.graphics_thread.plot_trajectories()
        else:
            raise RuntimeError(
                "Graphics thread not initialized. Cannot plot trajectories."
            )

    def set_interactive_mode(self):
        """Enable interactive mode for user-controlled mic and source positions.

        Initializes graphics context for interactive mouse-controlled positioning.
        Cannot be called after trajectories mode is set.

        Raises:
            RuntimeError: If mode already set via set_trajectories_mode().
        """
        if self.is_mode_set:
            raise RuntimeError(
                "Mode already set. Cannot set interactive mode after trajectories mode. "
                "Please use reset_all() to reset the context if you want to change the get_position method."
            )
        graphics_context = GraphicsContextManager(
            self.reverb.room, disable_interactive=False
        )
        self.graphics_thread = GraphicsThread(
            graphics_context,
            synchro_flag=self.synchro_flag,
            stop_all_flag=self.stop_all_flag,
        )
        self.swft_thread.set_interactive_mode(graphics_context)
        self.is_mode_set = True

    def set_trajectories_mode(
        self,
        dynamic_mic_pos: np.typing.NDArray,
        dynamic_source_pos: np.typing.NDArray,
        display_graphics: bool = False,
    ):
        """Enable trajectory mode with predefined mic and source positions.

        Cycles through provided positions for automated acoustic simulations.
        Optionally displays graphics as positions update. Cannot be called after
        interactive mode is set.

        Args:
            dynamic_mic_pos: Array of shape (n_positions, 3) with mic positions.
            dynamic_source_pos: Array of shape (n_positions, 3) with source positions.
            display_graphics: If True, visualize position updates. Defaults to False.

        Raises:
            RuntimeError: If mode already set via set_interactive_mode().
        """
        if self.is_mode_set:
            raise RuntimeError(
                "Mode already set. Cannot set trajectories mode after interactive mode. "
                "Please use reset_all() to reset the context if you want to change the get_position method."
            )
        if display_graphics:
            graphics_context = GraphicsContextManager(
                self.reverb.room, disable_interactive=True
            )
            self.graphics_thread = GraphicsThread(
                graphics_context,
                dynamic_mic_pos=torch.from_numpy(dynamic_mic_pos),
                dynamic_source_pos=torch.from_numpy(dynamic_source_pos),
                synchro_flag=self.synchro_flag,
                stop_all_flag=self.stop_all_flag,
            )
            self.swft_thread.set_interactive_mode(graphics_context)
            self.is_mode_set = True
        else:
            self.swft_thread.set_trajectories_mode(dynamic_mic_pos, dynamic_source_pos)
            self.is_mode_set = True

    def set_output_stream(self):
        """Initialize sounddevice output stream for real-time audio playback.

        Creates a callback-based audio stream that pulls processed buffers from
        the SWFT processing queue. Requires sounddevice package.

        Warns:
            UserWarning: If sounddevice library is not installed.
        """
        try:
            from sounddevice import OutputStream

            def reverb_callback(outdata, frames, time, status):
                if status:
                    print(status)
                if len(self.swft_thread.swft_context.output_queue) > 0:
                    outdata[:] = self.swft_thread.get_next_output_buffer()
                else:
                    print("Output queue is empty")

            self.output_stream = OutputStream(
                samplerate=self.audio_sr,
                channels=self.n_channels,
                blocksize=self.audio_buffer_size,
                dtype="float32",
                callback=reverb_callback,
            )
        except ImportError:
            warn("Sounddevice library not found.")

    def run_conductor(self):
        """Synchronization conductor thread that triggers buffer processing.

        Periodically sets synchronization flags to coordinate audio processing
        and graphics update threads at fixed time intervals matching buffer duration.
        """
        self.stop_flag = False
        eps = 0.0001
        while not self.stop_flag:
            self.synchro_flag.clear()
            sleep(self.audio_buffer_dur)
            self.synchro_flag.set()
            # sleep(eps)
        self.synchro_flag.clear()

    def stop_conductor(self):
        """Stop the synchronization conductor thread.

        Signals the conductor thread to exit its loop.
        """
        self.stop_flag = True

    def show_graphics(self):
        """Display the graphics window from graphics thread.

        Renders the current graphics output from the active graphics thread.

        Raises:
            RuntimeError: If graphics_thread not initialized.
        """
        if hasattr(self, "graphics_thread"):
            self.graphics_thread.show()
        else:
            raise RuntimeError("Graphics thread not initialized. Cannot show graphics.")

    def run(self) -> Tensor:
        """Execute real-time SWFT audio processing with optional graphics sync.

        Starts all worker threads (audio, graphics, conductor), synchronizes
        their execution, handles interrupts, and returns accumulated output
        if store_full_output was enabled.

        Returns:
            Tensor of accumulated output if store_full_output=True, else empty tensor.

        Raises:
            RuntimeError: If no position mode set (interactive or trajectories).
        """
        if not self.is_mode_set:
            raise RuntimeError(
                "No mode set for getting mic and source positions. "
                "Please call set_interactive_mode() or set_trajectories_mode() before running."
            )
        if self.swft_thread.store_output:
            warn(
                "store_full_output is set to True, which may cause high memory usage "
                "if the processing runs for a long time. Make sure to monitor memory usage "
                "or set store_full_output to False if you don't need the full output after processing."
            )
        try:
            self.swft_thread.start()
            if hasattr(self, "output_stream"):
                self.output_stream.__enter__()

            if hasattr(self, "graphics_thread"):
                self.graphics_thread.start()

            if hasattr(self, "graphics_thread") or hasattr(self, "output_stream"):
                print(
                    "Starting real-time processing with "
                    "synchronized graphics and/or audio output..."
                )
                self.conductor.start()
            else:
                # If no graphics or output stream,
                # just run the audio processing without synchronization
                self.synchro_flag.set()

            if hasattr(self, "graphics_thread"):
                self.graphics_thread.monitor_show()
            else:
                while self.swft_thread.inner_thread.is_alive():
                    sleep(0.0001)

        except (KeyboardInterrupt, StopIteration):
            print("Stopping real-time processing...")
        finally:
            if hasattr(self, "output_stream"):
                self.output_stream.__exit__(None, None, None)
            self.swft_thread.stop()
            self.swft_thread.join()
            if hasattr(self, "graphics_thread"):
                self.graphics_thread.stop()
                self.graphics_thread.join()

            if self.conductor.is_alive():
                self.stop_conductor()
                self.conductor.join()

        # Check if any exception was raised in the swft_thread and re-raise it in the main thread,
        # since exceptions in threads don't propagate to the main thread by default
        try:
            exc = self.swft_thread.error_bin.get(block=False)
            raise exc
        except (queue.Empty, KeyboardInterrupt, StopIteration):
            pass

        try:
            if hasattr(self, "graphics_thread"):
                exc = self.graphics_thread.error_bin.get(block=False)
                raise exc
        except (queue.Empty, KeyboardInterrupt, StopIteration):
            pass

        return (
            self.swft_thread.get_full_output()
            if self.swft_thread.store_output
            else torch.tensor([])
        )

    def reset_all(self):
        """Reset processor to initial state, stopping all threads and clearing state.

        Terminates conductor, graphics, and output stream threads if running.
        Clears all flags and resets internal state for fresh processing session.
        Position mode must be set again before calling run().
        """
        if self.conductor.is_alive():
            self.stop_conductor()
            self.conductor.join()
        del self.conductor
        self.conductor = threading.Thread(target=self.run_conductor)
        if hasattr(self, "graphics_thread"):
            if self.graphics_thread.inner_thread.is_alive():
                self.graphics_thread.stop()
                self.graphics_thread.join()
            del self.graphics_thread
        if hasattr(self, "output_stream"):
            if self.output_stream.active:
                self.output_stream.__exit__(None, None, None)
            del self.output_stream

        self.stop_all_flag.clear()
        # Pause the flag to prevent any threads from running until start() is called again
        self.synchro_flag.clear()
        self.swft_thread.reset_all()
        self.is_mode_set = False
