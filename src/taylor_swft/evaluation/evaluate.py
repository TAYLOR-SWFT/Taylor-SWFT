from ..room.datasets import BRASBenchmarkToSWFTRoom, BRASItemSWFTRoom
from ..utils.baselines import ALL_BASELINES, BASELINE_KWARGS
from .metrics import distance
from pathlib import Path
from time import perf_counter
from soundfile import write, read
from torch import from_numpy
from tqdm import tqdm
import hashlib
import pandas as pd


def get_hash(metadata: dict) -> int:
    """Generate a unique hash from metadata dictionary.

    Creates an MD5 hash from sorted metadata key-value pairs to provide a unique
    identifier for each RIR computation configuration.

    Args:
        metadata: Dictionary containing configuration parameters (scene, positions,
            baseline, etc.).

    Returns:
        int: 128-bit MD5 hash as an integer value.
    """
    strs = [f"{key}:{metadata[key]}" for key in sorted(metadata.keys())]
    combined_str = "|".join(strs)
    return int(hashlib.md5(combined_str.encode()).hexdigest(), 16)


def get_metadata(item: BRASItemSWFTRoom, baseline_name: str) -> dict:
    """Extract and augment metadata from a BRAS item for evaluation.

    Collects relevant metadata from the data item (scene name, positions, material
    count) and adds baseline-specific parameters and a unique hash identifier.

    Args:
        item: BRAS dataset item containing room, positions, and audio data.
        baseline_name: Name of the RIR synthesis baseline method.

    Returns:
        dict: Metadata dictionary with keys:
            - Scene/position info (scene_name, loudspeaker, microphone, etc.)
            - "n_faces" (int): Number of room surfaces.
            - "baseline" (str): Baseline method name.
            - "method_args" (dict): Baseline-specific keyword arguments.
            - "hash" (int): Unique configuration hash.
            - "rir_wav_name" (str): Output filename for the RIR.
    """
    metadata = {
        key: value
        for key, value in item.items()
        if isinstance(value, (str, int, float))
        or key in ["source_position", "receiver_position"]
    }
    metadata["n_faces"] = len(item["swft_room"].room.walls)
    metadata["baseline"] = baseline_name
    metadata["method_args"] = BASELINE_KWARGS.get(baseline_name, {})
    metadata["hash"] = get_hash(metadata)
    rir_name = f"{item['wav_file_name']}_{baseline_name}_{metadata['hash']}.wav"
    metadata["rir_wav_name"] = rir_name
    return metadata


def log_to_csv(csv_file: Path, row: dict) -> None:
    """Append a row to a CSV file, creating it if necessary.

    Args:
        csv_file: Path to the CSV file. Will be created if it doesn't exist.
        row: Dictionary representing a single row to append.
    """
    df = pd.DataFrame([row])
    df.to_csv(csv_file, mode="a", index=False, header=not csv_file.exists())


def clean_csv(csv_file: Path) -> None:
    """Remove duplicate rows from a CSV file based on hash column.

    Processes the CSV in reverse order (most recent entries first) and keeps
    only the first occurrence of each unique hash value. Saves cleaned output
    to a file with "_clean" suffix.

    Args:
        csv_file: Path to the CSV file to clean.

    Raises:
        FileNotFoundError: If the CSV file doesn't exist.
    """
    # Parse the CSV in reverse order and keep only the first occurrence of each hash
    if not csv_file.exists():
        return
    df = pd.read_csv(csv_file)
    n_to_drop = len(df)
    seen_hashes = set()
    for index in reversed(df.index):
        hash_value = df.at[index, "hash"]
        if hash_value in seen_hashes:
            df.drop(index, inplace=True)
        else:
            seen_hashes.add(hash_value)
    n_to_drop -= len(df)
    print(f"Removed {n_to_drop} duplicate rows from {csv_file}")
    clean_name = csv_file.stem + "_clean" + csv_file.suffix
    clean_path = csv_file.parent / clean_name
    df.to_csv(clean_path, index=False)


def evaluate(
    dataset: BRASBenchmarkToSWFTRoom,
    exp_dir: Path,
    recompute_metrics: bool,
) -> None:
    """Evaluate all baseline methods on the BRAS dataset.

    Computes RIRs using all registered baseline methods and saves audio files,
    metadata, and evaluation metrics to disk. Handles caching to avoid redundant
    computations when resuming from partial results.

    Args:
        dataset: The BRAS to SWFT dataset loader.
        exp_dir: Directory to save results (metadata.csv, results.csv, wavs/).
        recompute_metrics: If True, recompute metrics for cached RIRs.

    Raises:
        ValueError: If duplicate hashes are found in existing results.
        AssertionError: If sample rate mismatches occur.
    """
    exp_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = exp_dir / "wavs"
    wav_dir.mkdir(exist_ok=True)
    metadata_file = exp_dir / "metadata.csv"
    results_file = exp_dir / "results.csv"

    if metadata_file.exists():
        # Loading existing metadata
        metadata_df = pd.read_csv(metadata_file)
        existing_hashes = set(metadata_df["hash"])
        assert len(metadata_df) == len(existing_hashes), "Duplicate hashes found."
    else:
        # No metadata file -> First run
        existing_hashes = set()

    for item in tqdm(iter(dataset), total=len(dataset)):
        for baseline_name, baseline_func in ALL_BASELINES.items():
            metadata = get_metadata(item, baseline_name)

            if metadata["hash"] not in existing_hashes:
                # Hash not found -> Compute RIR
                try:
                    tik = perf_counter()
                    rir = baseline_func(
                        item["swft_room"],
                        item["source_position"],
                        item["receiver_position"],
                        **metadata["method_args"],
                    )
                    tok = perf_counter()
                except ValueError as e:
                    import warnings

                    # Sometimes ISM fails in CR1 and finds no visible source.
                    # The exception is uncaught in pra so we handle it here.
                    # Eventually it will run with the same args, the behaviour
                    # is non-deterministic. We just skip the sample for now.
                    warnings.warn(
                        f"Error computing RIR for {metadata['baseline']} in "
                        f"{metadata['scene_name']}: {e}"
                        "Re-compute the RIR with the same args."
                    )
                    continue

                # Save the RIR
                write(
                    wav_dir / metadata["rir_wav_name"],
                    rir.numpy(),
                    samplerate=item["sample_rate"],
                    subtype="FLOAT" if "32" in str(rir.dtype) else "DOUBLE",
                )

                # Save the computation time with the metadata
                metadata["computation_time"] = tok - tik
                log_to_csv(metadata_file, metadata)

                # Compute the metrics for the first time
                results = distance(
                    rir_ref=item["waveform"].squeeze(),
                    rir_other=rir.squeeze(),
                    sample_rate=item["sample_rate"],
                )
                results.update({"hash": metadata["hash"]})
                log_to_csv(results_file, results)
            elif recompute_metrics:
                # The RIR exists but we recompute the metrics
                # rir, sr_load = load(wav_dir / metadata["rir_wav_name"])
                rir, sr_load = read(wav_dir / metadata["rir_wav_name"])
                rir = from_numpy(rir.T)
                assert sr_load == item["sample_rate"], "Sample rate mismatch"
                results = distance(
                    rir_ref=item["waveform"].squeeze(),
                    rir_other=rir.squeeze(),
                    sample_rate=item["sample_rate"],
                )
                results.update({"hash": metadata["hash"]})
                log_to_csv(results_file, results)
            else:
                # The RIR exists and we skip metric computation
                continue
    clean_csv(results_file)


def run_BRAS_eval(
    data_dir: Path, exp_dir: Path, recompute_metrics: bool = False
) -> None:
    """Run a complete evaluation pipeline on the BRAS Benchmark dataset.

    Configures pyroomacoustics parameters, initializes the dataset, and evaluates
    all baseline methods.

    Args:
        data_dir: Path to the BRAS dataset directory.
        exp_dir: Output directory for results.
        recompute_metrics: If True, recompute metrics for all samples.
            Defaults to False.
    """
    from pyroomacoustics.parameters import constants

    constants.set("octave_bands_base_freq", 31.25)
    swft_dataset = BRASBenchmarkToSWFTRoom(
        base_data_path=data_dir,
        fs=32000,
        material_types="initial",
    )
    evaluate(swft_dataset, exp_dir, recompute_metrics)
