from taylor_swft import TaylorSWFTRealTimeProcessor, make_demo_room, Reverberator
import torchaudio
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


def parser():

    parser = argparse.ArgumentParser(
        description="Run the Taylor-SWFT interactive demo."
    )
    parser.add_argument(
        "--buffer-dur",
        type=float,
        default=30e-3,
        help="Duration of the audio buffer in seconds (default: 30e-3)",
    )
    parser.add_argument(
        "--interactive-mode",
        action="store_true",
        help="Whether to enable interactive mode with GUI (default: False). If not set, the demo will run with predefined trajectories for the microphone and source.",
    )
    parser.add_argument(
        "--reflection-order",
        type=int,
        default=4,
        help="Maximum order of reflections to consider for the early echoes (default: 4). Higher values will increase realism but also computational load.",
    )
    return parser.parse_args()


def interactive_demo(buffer_dur, interactive_mode, reflection_order):
    """Demonstration of real-time SWFT processing with dynamic position trajectories.

    Loads demo audio, creates circular trajectories for source and microphone,
    and processes the audio in real-time with graphics visualization.
    """

    # Create the SWFTRoom
    swft_room = make_demo_room()

    # Define global parameters
    BUFFER_DUR = buffer_dur
    N_CHANNELS = 1
    SAMPLE_RATE = swft_room.room.fs
    BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_DUR)
    OUTPUT_DIR = "./data/outputs"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Create the Reverberator from the SWFTRoom
    reverb = Reverberator(swft_room)

    # Load and preprocess the input audio
    input_audio, original_sr = torchaudio.load(
        "data/examples/demo_sound.flac", normalize=True, channels_first=False
    )
    if input_audio.shape[0] > N_CHANNELS:
        input_audio = input_audio[..., :N_CHANNELS]
    if original_sr != SAMPLE_RATE:
        input_audio = torchaudio.functional.resample(
            input_audio.T, orig_freq=original_sr, new_freq=SAMPLE_RATE
        ).T
    n_buffers = int(input_audio.shape[0] / BUFFER_SIZE)

    output = torch.zeros(n_buffers * BUFFER_SIZE)

    real_time_processor = TaylorSWFTRealTimeProcessor(
        reverb=reverb,
        input_audio=input_audio,
        audio_buffer_dur=BUFFER_DUR,
        n_channels=N_CHANNELS,
        reflection_order=reflection_order,
        overlap=True,
        device="cpu",
        store_full_output=True,
        loop_input_audio=True,
    )

    if interactive_mode:
        real_time_processor.set_interactive_mode()
    else:

        # Define dynamic trajectories for the microphone and source (two circles in this example)
        circle = lambda t, center, radius: torch.stack(
            [
                center[0] + radius * torch.cos(t),
                center[1] + radius * torch.sin(t),
                center[2] * torch.ones_like(t),
            ],
            dim=-1,
        )
        t_mic = torch.linspace(0, 4 * torch.pi, steps=n_buffers)
        dynamic_mic_pos = circle(t_mic, torch.tensor([30, 30, 15]), torch.tensor([10]))
        t_source = torch.linspace(0, 2 * torch.pi, steps=n_buffers) + torch.pi / 4
        dynamic_source_pos = circle(
            t_source, torch.tensor([45, 30, 15]), torch.tensor([5])
        )

        # Set the trajectories mode with the defined dynamic positions
        real_time_processor.set_trajectories_mode(
            dynamic_mic_pos=(
                dynamic_mic_pos.numpy() if len(dynamic_mic_pos) > 0 else np.array([])
            ),
            dynamic_source_pos=(
                dynamic_source_pos.numpy()
                if len(dynamic_source_pos) > 0
                else np.array([])
            ),
            display_graphics=True,
        )

    # Set the output stream
    real_time_processor.set_output_stream()

    # Run the real-time processing loop until the user stops it
    output = real_time_processor.run()

    # When the processing is stopped, plot the trajectories if the GUI was enabled
    if hasattr(real_time_processor, "graphics_thread"):
        real_time_processor.plot_trajectories()
        plt.savefig(f"{OUTPUT_DIR}/trajectories.png", dpi=300)

    # Normalize input and output to prevent clipping when saving or plotting
    output = output / output.abs().max() * 0.9
    input_audio = input_audio / input_audio.abs().max() * 0.9

    # Save the processed audio
    torchaudio.save(
        f"{OUTPUT_DIR}/reverberated_real_time.wav", output.unsqueeze(0), SAMPLE_RATE
    )

    # Plot and save the spectrograms of the input and output audio
    fig, axs = plt.subplots(2, 1, sharex=False, figsize=(10, 6))
    fontsize = 15

    plt.rcParams.update({"font.size": fontsize})
    axs[0].specgram(
        input_audio.squeeze().numpy(), Fs=SAMPLE_RATE, cmap="plasma", vmin=-140
    )
    axs[0].set_title("Input Audio Spectrogram")
    axs[0].set_ylabel("Frequency (Hz)", fontsize=int(fontsize * 0.8))
    axs[1].specgram(
        output.squeeze().numpy(),
        Fs=SAMPLE_RATE,
        cmap="plasma",
        vmin=-140,
    )
    axs[1].set_title("Output Audio Spectrogram")
    axs[1].set_ylabel("Frequency (Hz)", fontsize=int(fontsize * 0.8))
    plt.xlabel("Time (s)", fontsize=int(fontsize * 0.8))
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/spectrograms.png", dpi=300)


if __name__ == "__main__":
    args = parser()
    interactive_demo(
        buffer_dur=args.buffer_dur,
        interactive_mode=args.interactive_mode,
        reflection_order=args.reflection_order,
    )
