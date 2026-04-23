from .metrics import ALL_METRICS
from pathlib import Path
import numpy as np
import pandas as pd


def process_results(results_df: pd.DataFrame, exp_dir: Path) -> None:
    """Compute statistics from evaluation results and save per-baseline/room reports.

    Aggregates metric results by room and baseline method, computing mean, median,
    and standard deviation. Identifies best, second-best, and worst performing
    methods for each metric and room. Saves results to separate CSV files.

    Args:
        results_df: DataFrame containing evaluation results with columns including
            scene_name, baseline, and metric columns.
        exp_dir: Output directory where statistics CSVs will be saved.

    Raises:
        AssertionError: If results contain missing values or duplicate hashes.
    """
    # Check the integrity of the data
    print("Checking data integrity.")
    # No missing values
    assert not results_df.isnull().values.any(), "Some values are missing.)"
    # Hash values are unique
    assert results_df["hash"].is_unique, (
        "Some metrics were computed multiple times with "
        "the same data. Please run the clean_csv function"
        " from the evaluate module to remove duplicates."
    )

    # Create a new column for the room name
    # derived from the scene name
    results_df["room_name"] = results_df["scene_name"].apply(lambda x: x.split("_")[0])
    stat_dir = exp_dir / "stats"
    stat_dir.mkdir(exist_ok=True)
    metrics = list(ALL_METRICS.keys()) + ["computation_time"]
    results_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    stats = results_df.groupby(["room_name", "baseline"])[metrics].agg(
        ["mean", "median", "std"]
    )
    for metric in metrics:
        # Find the baseline with the minimum mean for each room
        best_indices = stats[(metric, "mean")].groupby("room_name").idxmin()
        second_best_indices = (
            stats[(metric, "mean")]
            .groupby("room_name")
            .apply(lambda x: x.nsmallest(2).index[-1])
        )
        worst_indices = stats[(metric, "mean")].groupby("room_name").idxmax()
        stats[(metric, "best")] = False
        stats[(metric, "second_best")] = False
        stats[(metric, "worst")] = False
        stats.loc[best_indices, (metric, "best")] = True
        stats.loc[second_best_indices, (metric, "second_best")] = True
        stats.loc[worst_indices, (metric, "worst")] = True

    for (room, baseline), group in stats.groupby(["room_name", "baseline"]):
        # Create a clean DataFrame for this specific pair
        output_rows = []

        for metric in metrics:
            output_rows.append(
                {
                    "metric": metric,
                    "mean": group[(metric, "mean")].values[0],
                    "std": group[(metric, "std")].values[0],
                    "is_best": group[(metric, "best")].values[0],
                    "is_worst": group[(metric, "worst")].values[0],
                    "is_second_best": group[(metric, "second_best")].values[0],
                    "median": group[(metric, "median")].values[0],
                }
            )

        pair_df = pd.DataFrame(output_rows)
        print(f"Saving stats for {baseline} in {room}.")
        pair_df.to_csv(stat_dir / f"stats_{baseline}_{room}.csv", index=False)


def save_statistics(exp_dir: Path) -> None:
    """Merge metadata with results and compute statistics for all baselines.

    Loads the cleaned results and metadata CSV files, merges them, and calls
    process_results to generate per-baseline/room statistics.

    Args:
        exp_dir: Experiment directory containing metadata.csv and results_clean.csv.

    Raises:
        FileNotFoundError: If required CSV files are not found.
        AssertionError: If required files are missing.
    """
    metadata_file = exp_dir / "metadata.csv"
    assert metadata_file.exists(), (
        f"Metadata file not found at {metadata_file}",
        "Please run an evaluation.",
    )
    result_file = exp_dir / "results_clean.csv"
    assert result_file.exists(), (
        f"Cleaned results file not found at {result_file}",
        "Please re-run the evaluation or run the clean_csv function.",
    )
    results_df = pd.read_csv(result_file)
    metadata_df = pd.read_csv(metadata_file)

    results_df = results_df.merge(metadata_df, on="hash", how="left")
    process_results(results_df, exp_dir)


def make_table(exp_dir: Path) -> None:
    """Generate a PDF table exactly as in the paper using typst compiler.

    Args:
        exp_dir: Experiment directory containing stats subdirectory.

    Raises:
        ImportError: If Typst is not installed.
        FileNotFoundError: If Typst template or stats directory is not found.
    """
    from taylor_swft.utils.constants import TYPST_PATH

    try:
        import typst
    except ImportError:
        raise ImportError("Typst is not installed. Please install it.")

    stat_dir = exp_dir / "stats"
    typst_file = Path(TYPST_PATH)
    assert typst_file.exists(), f"Typst template not found at {typst_file}"
    output_file = exp_dir / "results_table.pdf"
    root = stat_dir.parent

    with open(typst_file, "rb") as f:
        compiler = typst.Compiler(input=f.read(), root=root)
    compiler.compile(output=output_file)
