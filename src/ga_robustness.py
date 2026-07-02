import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from evaluator import get_order_economic_value, get_order_kg
from evaluator import get_available_line_time_for_day
from generate_instance import load_real_instance
from geneticalgorithm import run_genetic_algorithm


DEFAULT_OPERATIONAL_CONFIG = {
    "shift_start_min": 480,
    "shift_end_min": 990,
    "lunch_break_min": 30,
    "cleaning_time_min": 30,
}


def get_total_order_value_and_kg(instance):
    refs_by_id = {
        str(ref["ref_id"]).strip(): ref
        for ref in instance.get("references", [])
    }
    total_kg = 0.0
    total_value = 0.0

    for order in instance.get("demand", []):
        ref = refs_by_id.get(str(order.get("ref_id", "")).strip())
        if not ref:
            continue

        total_kg += get_order_kg(order, ref)
        total_value += get_order_economic_value(order, ref)

    return total_kg, total_value


def metric_row(seed, metrics, generations, instance):
    max_operator_minutes = (
        metrics.get("max_values", {}).get("operator_minutes", 0) or 0
    )
    operator_utilisation_pct = (
        metrics.get("operator_usage_minutes", 0) / max_operator_minutes * 100
        if max_operator_minutes
        else 0
    )
    capacity_l1_pct = line_capacity_utilisation_pct(metrics, instance, "L1")
    capacity_l2_pct = line_capacity_utilisation_pct(metrics, instance, "L2")

    return {
        "seed": seed,
        "generations": generations,
        "Z": metrics.get("normalised_fitness"),
        "scheduled_kg": metrics.get("scheduled_kg", 0),
        "postponed_kg": metrics.get("postponed_kg", 0),
        "scheduled_economic_value": metrics.get("scheduled_economic_value", 0),
        "postponed_economic_value": metrics.get("postponed_economic_value", 0),
        "setup_total_min": metrics.get("setup_total_min", 0),
        "capacity_utilisation_pct": (
            metrics.get("capacity_utilisation_ratio", 0) * 100
        ),
        "capacity_utilisation_L1_pct": capacity_l1_pct,
        "capacity_utilisation_L2_pct": capacity_l2_pct,
        "operator_utilisation_pct": operator_utilisation_pct,
        "operator_usage_minutes": metrics.get("operator_usage_minutes", 0),
        "peak_operators": metrics.get("peak_operators", 0),
        "standard_operators": metrics.get("standard_operators", 0),
        "max_daily_required_operators": max(
            metrics.get("operators_required_by_day", {0: 0}).values()
            or [0]
        ),
        "postponed_orders": metrics.get("postponed_orders", 0),
        "postponed_boxes": metrics.get("postponed_boxes", 0),
        "delay_days_total": metrics.get("delay_days_total", 0),
        "capacity_violations": metrics.get("capacity_violations", 0),
        "operator_violations": metrics.get("operator_violations", 0),
        "total_capacity_excess": metrics.get("total_capacity_excess", 0),
        "total_operator_excess": metrics.get("total_operator_excess", 0),
        "infeasible_solution": metrics.get("infeasible_solution", False),
    }


def line_capacity_utilisation_pct(metrics, instance, line):
    production_by_day_line = metrics.get("production_time_by_day_line", {})
    setup_by_day_line = metrics.get("setup_time_by_day_line", {})
    days = {
        day
        for day, current_line in production_by_day_line.keys()
        if current_line == line
    }
    days.update({
        day
        for day, current_line in setup_by_day_line.keys()
        if current_line == line
    })

    if not days:
        return 0.0

    values = []
    for day in sorted(days):
        available = get_available_line_time_for_day(instance, day)
        if not available:
            continue
        total_time = (
            production_by_day_line.get((day, line), 0)
            + setup_by_day_line.get((day, line), 0)
        )
        values.append(total_time / available * 100)

    return sum(values) / len(values) if values else 0.0


def sanity_row(row, total_kg, total_value):
    tolerance = 1e-6
    return {
        "seed": row["seed"],
        "euros_within_total": (
            row["scheduled_economic_value"] <= total_value + tolerance
        ),
        "kg_within_total": row["scheduled_kg"] <= total_kg + tolerance,
        "capacity_respected": (
            row["capacity_violations"] == 0
            and row["total_capacity_excess"] <= tolerance
        ),
        "operators_respected": (
            row["operator_violations"] == 0
            and row["total_operator_excess"] <= tolerance
        ),
    }


def summary_rows(raw_df):
    metric_columns = [
        "Z",
        "scheduled_kg",
        "postponed_kg",
        "scheduled_economic_value",
        "postponed_economic_value",
        "setup_total_min",
        "capacity_utilisation_pct",
        "capacity_utilisation_L1_pct",
        "capacity_utilisation_L2_pct",
        "operator_utilisation_pct",
        "operator_usage_minutes",
        "peak_operators",
        "standard_operators",
        "max_daily_required_operators",
        "postponed_orders",
        "postponed_boxes",
        "delay_days_total",
        "generations",
    ]

    rows = []
    for metric in metric_columns:
        series = pd.to_numeric(raw_df[metric], errors="coerce")
        rows.append({
            "metric": metric,
            "mean": series.mean(),
            "std": series.std(ddof=1) if len(series) > 1 else 0.0,
            "min": series.min(),
            "max": series.max(),
        })
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate_figures(raw_df, convergence_df, output_dir):
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.hist(raw_df["Z"], bins=min(12, max(5, len(raw_df) // 3)),
             color="#1f77b4", edgecolor="white")
    plt.axvline(raw_df["Z"].mean(), color="#c0003b", linestyle="--",
                label=f"Mean Z = {raw_df['Z'].mean():.4f}")
    plt.title("GA robustness - distribution of final fitness")
    plt.xlabel("Final normalised fitness Z")
    plt.ylabel("Number of runs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "z_distribution.png", dpi=200)
    plt.close()

    pivot = convergence_df.pivot_table(
        index="seed",
        columns="generation",
        values="best_fitness",
        aggfunc="first",
    )
    plt.figure(figsize=(12, 5))
    plt.imshow(pivot, aspect="auto", cmap="Blues_r")
    plt.colorbar(label="Best fitness Z")
    plt.title("GA robustness - convergence heatmap by seed")
    plt.xlabel("Generation")
    plt.ylabel("Seed")
    plt.yticks(
        ticks=range(len(pivot.index)),
        labels=[str(seed) for seed in pivot.index],
    )
    plt.tight_layout()
    plt.savefig(plots_dir / "convergence_heatmap.png", dpi=200)
    plt.close()

    grouped = convergence_df.groupby("generation")["best_fitness"]
    mean_curve = grouped.mean()
    min_curve = grouped.min()
    max_curve = grouped.max()

    plt.figure(figsize=(10, 5))
    plt.plot(mean_curve.index, mean_curve.values, color="#1f77b4",
             label="Mean best fitness")
    plt.fill_between(mean_curve.index, min_curve.values, max_curve.values,
                     color="#1f77b4", alpha=0.20, label="Min-max range")
    plt.title("GA robustness - mean convergence across seeds")
    plt.xlabel("Generation")
    plt.ylabel("Best fitness Z")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "mean_convergence.png", dpi=200)
    plt.close()


def run_single(args):
    instance = load_real_instance(
        str(args.input),
        operational_config=DEFAULT_OPERATIONAL_CONFIG,
    )

    result = run_genetic_algorithm(
        instance,
        population_size=args.population_size,
        mutation_rate=args.mutation_rate,
        stagnation_k=args.stagnation_k,
        generations=args.generations,
        elite_size=args.elite_size,
        tournament_size=args.tournament_size,
        seed=args.seed,
        heuristic_ratio=args.heuristic_ratio,
        verbose=False,
        return_history=True,
    )
    _, metrics, generations, history = result
    row = metric_row(args.seed, metrics, generations, instance)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / f"ga_robustness_seed_{args.seed}.csv", [row])

    history_rows = [
        {
            "seed": args.seed,
            "generation": item["generation"],
            "best_fitness": item["best_fitness"],
        }
        for item in history
    ]
    write_csv(
        args.output_dir / f"ga_robustness_history_seed_{args.seed}.csv",
        history_rows,
    )

    print("=== GA ROBUSTNESS RUN ===")
    for key, value in row.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


def collect_results(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_files = sorted(args.output_dir.glob("**/ga_robustness_seed_*.csv"))
    history_files = sorted(
        args.output_dir.glob("**/ga_robustness_history_seed_*.csv")
    )

    if not metric_files:
        raise FileNotFoundError(
            f"No ga_robustness_seed_*.csv files found in {args.output_dir}"
        )

    raw_df = pd.concat(
        [pd.read_csv(path) for path in metric_files],
        ignore_index=True,
    ).drop_duplicates(subset=["seed"], keep="last")
    raw_df = raw_df.sort_values("seed")

    convergence_df = pd.concat(
        [pd.read_csv(path) for path in history_files],
        ignore_index=True,
    ).drop_duplicates(subset=["seed", "generation"], keep="last")
    convergence_df = convergence_df.sort_values(["seed", "generation"])

    raw_df.to_csv(
        args.output_dir / "ga_robustness_raw_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    convergence_df.to_csv(
        args.output_dir / "ga_robustness_convergence_history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = pd.DataFrame(summary_rows(raw_df))
    summary.to_csv(
        args.output_dir / "ga_robustness_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    instance = load_real_instance(
        str(args.input),
        operational_config=DEFAULT_OPERATIONAL_CONFIG,
    )
    total_kg, total_value = get_total_order_value_and_kg(instance)
    sanity = pd.DataFrame(
        [sanity_row(row, total_kg, total_value)
         for row in raw_df.to_dict("records")]
    )
    sanity.to_csv(
        args.output_dir / "ga_robustness_sanity_checks.csv",
        index=False,
        encoding="utf-8-sig",
    )

    generate_figures(raw_df, convergence_df, args.output_dir)

    try:
        with pd.ExcelWriter(args.output_dir / "ga_robustness_results.xlsx") as writer:
            raw_df.to_excel(writer, sheet_name="raw_runs", index=False)
            summary.to_excel(writer, sheet_name="summary", index=False)
            sanity.to_excel(writer, sheet_name="sanity_checks", index=False)
            convergence_df.to_excel(
                writer,
                sheet_name="convergence_history",
                index=False,
            )
    except ImportError:
        print("openpyxl not available; Excel file was not generated.")

    print("=== GA ROBUSTNESS SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== SANITY CHECKS ===")
    for column in [
        "euros_within_total",
        "kg_within_total",
        "capacity_respected",
        "operators_respected",
    ]:
        violations = int((~sanity[column]).sum())
        print(f"{column}: {violations} violations in {len(sanity)} runs")


def parse_args():
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    parser = argparse.ArgumentParser(
        description="Run or collect GA robustness experiments."
    )
    parser.add_argument("--input", default=str(repo_dir / "Inputs_June.xlsx"))
    parser.add_argument("--output-dir", type=Path,
                        default=repo_dir / "outputs" / "ga_robustness")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--population-size", type=int, default=108)
    parser.add_argument("--mutation-rate", type=float, default=0.057)
    parser.add_argument("--stagnation-k", type=int, default=26)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--elite-size", type=int, default=5)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--heuristic-ratio", type=float, default=0.15)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.collect_only:
        collect_results(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
