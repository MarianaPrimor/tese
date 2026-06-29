import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from generate_instance import load_real_instance
from geneticalgorithm import run_genetic_algorithm


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_INSTANCE_FILE = PROJECT_DIR / "Inputs_July.xlsx"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "edd_initialization_comparison"

GA_PARAMETERS = {
    "population_size": 108,
    "generations": 200,
    "mutation_rate": 0.057,
    "elite_size": 5,
    "tournament_size": 3,
    "stagnation_k": 26,
}
SEEDS = [42, 43, 44]
CONFIGURATIONS = [
    ("random_0", 0.0),
    ("edd_25", 0.25),
    ("edd_50", 0.50),
    ("edd_100", 1.0),
]
CONFIGURATION_BY_RATIO = {
    ratio: configuration
    for configuration, ratio in CONFIGURATIONS
}
OPERATIONAL_CONFIG = {
    "shift_start_min": 480,
    "shift_end_min": 990,
    "lunch_break_min": 30,
    "cleaning_time_min": 30,
    "standard_operators": 20,
    "operators": 20,
}


def load_instance(instance_file):
    return load_real_instance(
        str(instance_file),
        operational_config=OPERATIONAL_CONFIG,
    )


def first_best_generation(history):
    best_value = min(row["best_fitness"] for row in history)
    for row in history:
        if row["best_fitness"] == best_value:
            return row["generation"]
    return history[-1]["generation"]


def configuration_name(heuristic_ratio):
    return CONFIGURATION_BY_RATIO.get(
        heuristic_ratio,
        f"edd_{int(round(heuristic_ratio * 100))}",
    )


def run_single_configuration(instance, output_dir, heuristic_ratio, seed):
    configuration = configuration_name(heuristic_ratio)
    _solution, metrics, actual_generations, history = run_genetic_algorithm(
        instance,
        seed=seed,
        heuristic_ratio=heuristic_ratio,
        return_history=True,
        verbose=False,
        **GA_PARAMETERS,
    )

    raw_df = pd.DataFrame([
        {
            "configuration": configuration,
            "heuristic_ratio": heuristic_ratio,
            "seed": seed,
            "generation": row["generation"],
            "best_fitness": row["best_fitness"],
        }
        for row in history
    ])
    seed_summary_df = pd.DataFrame([{
        "configuration": configuration,
        "heuristic_ratio": heuristic_ratio,
        "seed": seed,
        "final_best_fitness": metrics["normalised_fitness"],
        "first_best_generation": first_best_generation(history),
        "total_generations_run": actual_generations,
    }])

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{configuration}_seed_{seed}"
    raw_df.to_csv(
        output_dir / f"edd_initialization_convergence_raw_{suffix}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    seed_summary_df.to_csv(
        output_dir / f"edd_initialization_seed_summary_{suffix}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return raw_df, seed_summary_df


def summarize_seed_results(seed_summary_df):
    return (
        seed_summary_df
        .groupby(["configuration", "heuristic_ratio"], as_index=False)
        .agg(
            mean_final_best_fitness=("final_best_fitness", "mean"),
            std_final_best_fitness=("final_best_fitness", "std"),
            mean_first_best_generation=("first_best_generation", "mean"),
            mean_total_generations_run=("total_generations_run", "mean"),
        )
    )


def run_comparison(instance, output_dir):
    raw_rows = []
    summary_seed_rows = []

    for configuration, heuristic_ratio in CONFIGURATIONS:
        for seed in SEEDS:
            _solution, metrics, actual_generations, history = run_genetic_algorithm(
                instance,
                seed=seed,
                heuristic_ratio=heuristic_ratio,
                return_history=True,
                verbose=False,
                **GA_PARAMETERS,
            )

            for history_row in history:
                raw_rows.append({
                    "configuration": configuration,
                    "heuristic_ratio": heuristic_ratio,
                    "seed": seed,
                    "generation": history_row["generation"],
                    "best_fitness": history_row["best_fitness"],
                })

            summary_seed_rows.append({
                "configuration": configuration,
                "heuristic_ratio": heuristic_ratio,
                "seed": seed,
                "final_best_fitness": metrics["normalised_fitness"],
                "first_best_generation": first_best_generation(history),
                "total_generations_run": actual_generations,
            })

    raw_df = pd.DataFrame(raw_rows)
    seed_summary_df = pd.DataFrame(summary_seed_rows)
    summary_df = summarize_seed_results(seed_summary_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(
        output_dir / "edd_initialization_convergence_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_df.to_csv(
        output_dir / "edd_initialization_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    seed_summary_df.to_csv(
        output_dir / "edd_initialization_seed_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return raw_df, summary_df, seed_summary_df


def collect_existing_outputs(output_dir):
    raw_path = output_dir / "edd_initialization_convergence_raw.csv"
    seed_path = output_dir / "edd_initialization_seed_summary.csv"

    if raw_path.exists() and seed_path.exists():
        raw_df = pd.read_csv(raw_path)
        seed_summary_df = pd.read_csv(seed_path)
    else:
        raw_frames = [
            pd.read_csv(path)
            for path in sorted(output_dir.glob("edd_initialization_convergence_raw_*.csv"))
        ]
        seed_frames = [
            pd.read_csv(path)
            for path in sorted(output_dir.glob("edd_initialization_seed_summary_*.csv"))
        ]

        if not raw_frames or not seed_frames:
            raise FileNotFoundError(
                f"No EDD initialization CSV outputs found in {output_dir}"
            )

        raw_df = pd.concat(raw_frames, ignore_index=True)
        seed_summary_df = pd.concat(seed_frames, ignore_index=True)

    summary_df = summarize_seed_results(seed_summary_df)
    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    seed_summary_df.to_csv(seed_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(
        output_dir / "edd_initialization_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return raw_df, summary_df, seed_summary_df


def make_full_generation_grid(raw_df):
    rows = []
    max_generation = int(raw_df["generation"].max())

    for (configuration, heuristic_ratio, seed), group in raw_df.groupby(
        ["configuration", "heuristic_ratio", "seed"]
    ):
        group = group.sort_values("generation")
        best_by_generation = dict(zip(group["generation"], group["best_fitness"]))
        last_value = None

        for generation in range(max_generation + 1):
            if generation in best_by_generation:
                last_value = best_by_generation[generation]
            if last_value is None:
                continue
            rows.append({
                "configuration": configuration,
                "heuristic_ratio": heuristic_ratio,
                "seed": seed,
                "generation": generation,
                "best_fitness": last_value,
            })

    return pd.DataFrame(rows)


def plot_convergence(raw_df, output_dir):
    full_df = make_full_generation_grid(raw_df)
    grouped = (
        full_df
        .groupby(["configuration", "heuristic_ratio", "generation"])["best_fitness"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(["heuristic_ratio", "generation"])
    )

    fig, ax = plt.subplots(figsize=(10, 5.8))
    colors = {
        "random_0": "#153e7e",
        "edd_25": "#2f7d32",
        "edd_50": "#f59e0b",
        "edd_100": "#b6003b",
    }

    for configuration, config_df in grouped.groupby("configuration"):
        config_df = config_df.sort_values("generation")
        x = config_df["generation"].to_numpy()
        y = config_df["mean"].to_numpy()
        std = config_df["std"].fillna(0).to_numpy()
        label = configuration.replace("_", " ")
        color = colors.get(configuration, "#172033")

        ax.plot(x, y, linewidth=2.2, color=color, label=label)
        ax.fill_between(x, y - std, y + std, color=color, alpha=0.16)

    ax.set_title("Convergence comparison by initialization strategy", fontweight="bold")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Configuration", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "convergence_comparison.png", dpi=180)
    plt.close(fig)


def plot_summary(seed_summary_df, output_dir):
    order = [configuration for configuration, _ in CONFIGURATIONS]
    labels = ["Random", "EDD 25%", "EDD 50%", "EDD 100%"]

    final_data = [
        seed_summary_df.loc[
            seed_summary_df["configuration"] == configuration,
            "final_best_fitness",
        ].tolist()
        for configuration in order
    ]
    gen_data = [
        seed_summary_df.loc[
            seed_summary_df["configuration"] == configuration,
            "first_best_generation",
        ].tolist()
        for configuration in order
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].boxplot(final_data, labels=labels, patch_artist=True)
    axes[0].set_title("Final best fitness by strategy", fontweight="bold")
    axes[0].set_ylabel("Final best fitness")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].boxplot(gen_data, labels=labels, patch_artist=True)
    axes[1].set_title("Generation where best fitness was reached", fontweight="bold")
    axes[1].set_ylabel("Generation")
    axes[1].grid(axis="y", alpha=0.25)

    for ax in axes:
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("EDD initialization impact summary", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_dir / "edd_initialization_summary_visual.png", dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare random and EDD-hybrid initialization strategies."
    )
    parser.add_argument("--instance-file", default=str(DEFAULT_INSTANCE_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--heuristic-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Combine existing CSV outputs and generate figures without running the GA.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)

    if args.plots_only:
        raw_df, _summary_df, seed_summary_df = collect_existing_outputs(output_dir)
        plot_convergence(raw_df, output_dir)
        plot_summary(seed_summary_df, output_dir)
        print(f"Wrote EDD initialization comparison figures to {output_dir}")
        return

    instance = load_instance(pathlib.Path(args.instance_file))

    if args.heuristic_ratio is not None or args.seed is not None:
        if args.heuristic_ratio is None or args.seed is None:
            raise ValueError("--heuristic-ratio and --seed must be used together.")
        run_single_configuration(
            instance,
            output_dir,
            args.heuristic_ratio,
            args.seed,
        )
        print(f"Wrote single EDD initialization run outputs to {output_dir}")
        return
    else:
        raw_df, _summary_df, seed_summary_df = run_comparison(instance, output_dir)

    plot_convergence(raw_df, output_dir)
    plot_summary(seed_summary_df, output_dir)
    print(f"Wrote EDD initialization comparison outputs to {output_dir}")


if __name__ == "__main__":
    main()
