import argparse
import csv
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from evaluator import get_available_line_time_for_day
from generate_instance import load_real_instance
from geneticalgorithm import run_genetic_algorithm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

DEFAULT_INSTANCE_FILES = {
    "june": (REPO_DIR / "Inputs_June.xlsx").resolve(),
    "july": (REPO_DIR / "Inputs_July.xlsx").resolve(),
    "august": (REPO_DIR / "Inputs_August.xlsx").resolve(),
}

DEFAULT_RESULT_FILES = {
    "june": SCRIPT_DIR / "optuna_results_june_new_algo_30trials_v1.csv",
    "july": SCRIPT_DIR / "optuna_results_july_new_algo_30trials_v1.csv",
    "august": SCRIPT_DIR / "optuna_results_august_new_algo_30trials_v1.csv",
}

SEEDS = [42, 43, 44]
MAX_GENERATIONS = 200
OUTPUT_DIR = SCRIPT_DIR / "candidate_parameter_validation"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate best monthly Optuna parameter candidates across all instances."
    )
    parser.add_argument("--operators", type=int, default=20)
    parser.add_argument("--workers", type=int, default=27)
    parser.add_argument("--june-non-working-dates", default="")
    parser.add_argument("--july-non-working-dates", default="")
    parser.add_argument("--august-non-working-dates", default="")
    return parser.parse_args()


def split_dates(value):
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def build_operational_config(operators, non_working_dates):
    return {
        "shift_start_min": 480,
        "shift_end_min": 990,
        "lunch_break_min": 30,
        "cleaning_time_min": 30,
        "standard_operators": operators,
        "non_working_dates": non_working_dates,
    }


def load_best_candidate(source_month, path):
    df = pd.read_csv(path)
    if "state" in df.columns:
        df = df[df["state"].astype(str).str.upper() == "COMPLETE"]
    if df.empty:
        raise ValueError(f"No complete Optuna trials found in {path}")

    best = df.loc[df["value"].astype(float).idxmin()]
    return {
        "candidate": f"best_{source_month}",
        "source_month": source_month,
        "source_trial": int(best["number"]),
        "source_mean_fitness": float(best["value"]),
        "population_size": int(best["params_population_size"]),
        "mutation_rate": float(best["params_mutation_rate"]),
        "stagnation_k": int(best["params_stagnation_k"]),
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


def run_candidate_seed(candidate, instance_name, instance_file, operators, non_working_dates, seed):
    instance = load_real_instance(
        str(instance_file),
        operational_config=build_operational_config(operators, non_working_dates),
    )
    start_time = time.perf_counter()
    _, metrics, generations = run_genetic_algorithm(
        instance,
        population_size=candidate["population_size"],
        mutation_rate=candidate["mutation_rate"],
        stagnation_k=candidate["stagnation_k"],
        generations=MAX_GENERATIONS,
        elite_size=None,
        tournament_size=3,
        seed=seed,
        verbose=False,
    )
    elapsed_s = time.perf_counter() - start_time

    max_operator_minutes = metrics.get("max_values", {}).get("operator_minutes", 0) or 0
    operator_utilisation_pct = (
        metrics.get("operator_usage_minutes", 0) / max_operator_minutes * 100
        if max_operator_minutes
        else 0.0
    )

    return {
        "candidate": candidate["candidate"],
        "source_month": candidate["source_month"],
        "source_trial": candidate["source_trial"],
        "source_mean_fitness": candidate["source_mean_fitness"],
        "population_size": candidate["population_size"],
        "mutation_rate": candidate["mutation_rate"],
        "stagnation_k": candidate["stagnation_k"],
        "instance": instance_name,
        "seed": seed,
        "Z": metrics.get("normalised_fitness"),
        "generations": generations,
        "elapsed_s": elapsed_s,
        "scheduled_kg": metrics.get("scheduled_kg", 0),
        "postponed_kg": metrics.get("postponed_kg", 0),
        "scheduled_economic_value": metrics.get("scheduled_economic_value", 0),
        "postponed_economic_value": metrics.get("postponed_economic_value", 0),
        "setup_total_min": metrics.get("setup_total_min", 0),
        "capacity_utilisation_pct": metrics.get("capacity_utilisation_ratio", 0) * 100,
        "capacity_utilisation_L1_pct": line_capacity_utilisation_pct(metrics, instance, "L1"),
        "capacity_utilisation_L2_pct": line_capacity_utilisation_pct(metrics, instance, "L2"),
        "operator_utilisation_pct": operator_utilisation_pct,
        "operator_usage_minutes": metrics.get("operator_usage_minutes", 0),
        "peak_operators": metrics.get("peak_operators", 0),
        "standard_operators": metrics.get("standard_operators", operators),
        "postponed_orders": metrics.get("postponed_orders", 0),
        "postponed_boxes": metrics.get("postponed_boxes", 0),
        "delay_days_total": metrics.get("delay_days_total", 0),
        "capacity_violations": metrics.get("capacity_violations", 0),
        "operator_violations": metrics.get("operator_violations", 0),
        "total_capacity_excess": metrics.get("total_capacity_excess", 0),
        "total_operator_excess": metrics.get("total_operator_excess", 0),
        "infeasible_solution": metrics.get("infeasible_solution", False),
    }


def summarise(raw_df, group_cols):
    metrics = [
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
        "postponed_orders",
        "postponed_boxes",
        "delay_days_total",
        "generations",
    ]
    rows = []
    for keys, group in raw_df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        first = group.iloc[0]
        for col in ["population_size", "mutation_rate", "stagnation_k", "source_trial"]:
            if col in group.columns:
                row[col] = first[col]
        for metric in metrics:
            values = group[metric].dropna().astype(float).tolist()
            row[f"{metric}_mean"] = sum(values) / len(values) if values else 0.0
            row[f"{metric}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
            row[f"{metric}_min"] = min(values) if values else 0.0
            row[f"{metric}_max"] = max(values) if values else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    non_working_dates = {
        "june": split_dates(args.june_non_working_dates),
        "july": split_dates(args.july_non_working_dates),
        "august": split_dates(args.august_non_working_dates),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates = [
        load_best_candidate(month, DEFAULT_RESULT_FILES[month])
        for month in ["june", "july", "august"]
    ]

    print("Selected candidates:", flush=True)
    for candidate in candidates:
        print(candidate, flush=True)

    tasks = []
    for candidate in candidates:
        for instance_name, instance_file in DEFAULT_INSTANCE_FILES.items():
            for seed in SEEDS:
                tasks.append((
                    candidate,
                    instance_name,
                    instance_file,
                    args.operators,
                    non_working_dates[instance_name],
                    seed,
                ))

    raw_rows = []
    max_workers = max(1, min(args.workers, len(tasks)))
    print(f"Running {len(tasks)} candidate/instance/seed evaluations with {max_workers} workers.", flush=True)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(run_candidate_seed, *task): task
            for task in tasks
        }
        for future in as_completed(future_to_task):
            row = future.result()
            raw_rows.append(row)
            print(
                f"{row['candidate']} on {row['instance']} seed {row['seed']}: "
                f"Z={row['Z']:.8f}, generations={row['generations']}",
                flush=True,
            )

    raw_df = pd.DataFrame(raw_rows).sort_values(["candidate", "instance", "seed"])
    by_instance = summarise(raw_df, ["candidate", "instance"])
    global_summary = summarise(raw_df, ["candidate"])
    global_summary = global_summary.sort_values("Z_mean")
    global_summary["rank_by_mean"] = range(1, len(global_summary) + 1)
    global_summary["worst_instance_seed_Z"] = (
        raw_df.groupby("candidate")["Z"].max().reindex(global_summary["candidate"]).values
    )
    global_summary = global_summary.sort_values("worst_instance_seed_Z")
    global_summary["rank_by_minimax"] = range(1, len(global_summary) + 1)
    global_summary = global_summary.sort_values("rank_by_mean")

    raw_df.to_csv(OUTPUT_DIR / "candidate_validation_raw.csv", index=False)
    by_instance.to_csv(OUTPUT_DIR / "candidate_validation_by_instance.csv", index=False)
    global_summary.to_csv(OUTPUT_DIR / "candidate_validation_global_summary.csv", index=False)

    print("\nGlobal summary:", flush=True)
    print(global_summary[[
        "candidate",
        "population_size",
        "mutation_rate",
        "stagnation_k",
        "Z_mean",
        "Z_std",
        "Z_min",
        "Z_max",
        "worst_instance_seed_Z",
        "rank_by_mean",
        "rank_by_minimax",
    ]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
