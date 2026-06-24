import argparse
import csv
import statistics
import time
from pathlib import Path

import pandas as pd

from generate_instance import load_real_instance
from geneticalgorithm import run_genetic_algorithm


SCRIPT_DIR = Path(__file__).resolve().parent
INSTANCE_FILE = (SCRIPT_DIR / "../Inputs_EmpresaX.xlsx").resolve()
SEEDS = [0, 7, 13, 42, 99]
N_RUNS = len(SEEDS)
MAX_GENERATIONS = 200
OBJECTIVE_VERSION = "normalised_v3_capacity_utilisation"
CHECKPOINT_FILE = SCRIPT_DIR / "ofat_run_checkpoint_normalised.csv"

CHECKPOINT_FIELDS = [
    "objective_version",
    "instance_signature",
    "experiment",
    "param_name",
    "param_value",
    "population_size",
    "mutation_rate",
    "stagnation_k",
    "elite_size",
    "tournament_size",
    "max_generations",
    "seed",
    "fitness",
    "generations",
    "elapsed_s",
]

instance = load_real_instance(
    str(INSTANCE_FILE),
    operational_config={
        "shift_start_min": 480,
        "shift_end_min": 990,
        "lunch_break_min": 30,
        "cleaning_time_min": 30,
    },
)


def instance_signature():
    stat = INSTANCE_FILE.stat()
    return f"{INSTANCE_FILE.name}:{stat.st_size}:{stat.st_mtime_ns}"


INSTANCE_SIGNATURE = instance_signature()


def checkpoint_key(row):
    return (
        row["objective_version"],
        row["instance_signature"],
        row["experiment"],
        row["param_name"],
        str(row["param_value"]),
        int(row["population_size"]),
        float(row["mutation_rate"]),
        int(row["stagnation_k"]),
        int(row["elite_size"]),
        int(row["tournament_size"]),
        int(row["max_generations"]),
        int(row["seed"]),
    )


def load_checkpoints():
    if not CHECKPOINT_FILE.exists():
        return {}

    checkpoints = {}

    with CHECKPOINT_FILE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            checkpoints[checkpoint_key(row)] = row

    return checkpoints


checkpoints = load_checkpoints()


def save_checkpoints():
    temp_path = CHECKPOINT_FILE.with_suffix(".csv.tmp")

    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CHECKPOINT_FIELDS)
        writer.writeheader()
        writer.writerows(checkpoints.values())

    temp_path.replace(CHECKPOINT_FILE)


def atomic_save_summary(path, summary):
    if not summary:
        return

    path = SCRIPT_DIR / path
    temp_path = path.with_suffix(".csv.tmp")

    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    temp_path.replace(path)


def create_checkpoint_row(
    experiment,
    param_name,
    param_value,
    params,
    seed,
    fitness,
    generations,
    elapsed,
):
    return {
        "objective_version": OBJECTIVE_VERSION,
        "instance_signature": INSTANCE_SIGNATURE,
        "experiment": experiment,
        "param_name": param_name,
        "param_value": param_value,
        "population_size": params["population_size"],
        "mutation_rate": params["mutation_rate"],
        "stagnation_k": params["stagnation_k"],
        "elite_size": params["elite_size"],
        "tournament_size": params["tournament_size"],
        "max_generations": MAX_GENERATIONS,
        "seed": seed,
        "fitness": f"{fitness:.12f}",
        "generations": generations,
        "elapsed_s": f"{elapsed:.3f}",
    }


def run_batch(
    experiment,
    param_name,
    param_values,
    fixed_params,
    summary_file,
):
    summary = []

    for value in param_values:
        params = dict(fixed_params)
        params[param_name] = value
        fitnesses = []
        generations = []
        times = []

        print(f"\n[{param_name} = {value}]", flush=True)

        for run_index, seed in enumerate(SEEDS, start=1):
            lookup_row = create_checkpoint_row(
                experiment,
                param_name,
                value,
                params,
                seed,
                fitness=0,
                generations=0,
                elapsed=0,
            )
            key = checkpoint_key(lookup_row)
            cached = checkpoints.get(key)

            if cached is not None:
                fitness = float(cached["fitness"])
                actual_generations = int(cached["generations"])
                elapsed = float(cached["elapsed_s"])
                print(
                    f"  run {run_index}/{N_RUNS} | seed={seed} | "
                    f"checkpoint fitness={fitness:.8f}",
                    flush=True,
                )
            else:
                print(
                    f"  starting run {run_index}/{N_RUNS} | seed={seed}",
                    flush=True,
                )
                start_time = time.perf_counter()
                _, metrics, actual_generations = run_genetic_algorithm(
                    instance,
                    generations=MAX_GENERATIONS,
                    seed=seed,
                    verbose=False,
                    **params,
                )
                elapsed = time.perf_counter() - start_time
                fitness = metrics["normalised_fitness"]
                completed_row = create_checkpoint_row(
                    experiment,
                    param_name,
                    value,
                    params,
                    seed,
                    fitness,
                    actual_generations,
                    elapsed,
                )
                checkpoints[checkpoint_key(completed_row)] = completed_row
                save_checkpoints()
                print(
                    f"  run {run_index}/{N_RUNS} saved | "
                    f"fitness={fitness:.8f} | "
                    f"gens={actual_generations} | "
                    f"time={elapsed:.1f}s",
                    flush=True,
                )

            fitnesses.append(fitness)
            generations.append(actual_generations)
            times.append(elapsed)

        row = {
            param_name: value,
            "mean_fitness": round(statistics.mean(fitnesses), 8),
            "std_fitness": round(statistics.stdev(fitnesses), 8),
            "best_fitness": round(min(fitnesses), 8),
            "mean_generations": round(statistics.mean(generations), 1),
            "mean_time_s": round(statistics.mean(times), 2),
        }
        summary.append(row)
        atomic_save_summary(summary_file, summary)

        print(
            f"  summary saved | mean={statistics.mean(fitnesses):.8f} | "
            f"std={statistics.stdev(fitnesses):.8f} | "
            f"best={min(fitnesses):.8f}",
            flush=True,
        )

    return summary


def run_population_seed(population_size, seed, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    _, metrics, actual_generations = run_genetic_algorithm(
        instance,
        population_size=population_size,
        generations=MAX_GENERATIONS,
        mutation_rate=0.10,
        elite_size=5,
        tournament_size=3,
        stagnation_k=20,
        seed=seed,
        verbose=False,
    )
    elapsed = time.perf_counter() - start_time

    row = {
        "objective_version": OBJECTIVE_VERSION,
        "instance": INSTANCE_FILE.name,
        "population_size": population_size,
        "mutation_rate": 0.10,
        "stagnation_k": 20,
        "elite_size": 5,
        "tournament_size": 3,
        "max_generations": MAX_GENERATIONS,
        "seed": seed,
        "fitness": metrics["normalised_fitness"],
        "generations": actual_generations,
        "elapsed_s": elapsed,
        "postponed_orders": metrics.get("postponed_orders", 0),
        "postponed_boxes": metrics.get("postponed_boxes", 0),
        "delay_days": metrics.get("delay_days_total", 0),
        "setup_time_min": metrics.get("setup_total_min", 0),
        "economic_value": metrics.get("scheduled_economic_value", 0),
        "capacity_utilisation": metrics.get("capacity_utilisation_ratio", 0),
        "operator_minutes": metrics.get("operator_usage_minutes", 0),
    }

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)

    print(
        f"Saved {output_path} | population={population_size} | "
        f"seed={seed} | fitness={row['fitness']:.8f}",
        flush=True,
    )


def aggregate_population_results(input_dir, output_xlsx):
    input_dir = Path(input_dir)
    result_files = sorted(input_dir.rglob("population_*_seed_*.csv"))

    if not result_files:
        raise FileNotFoundError(
            f"No population result CSV files found under {input_dir}."
        )

    individual = pd.concat(
        [pd.read_csv(path) for path in result_files],
        ignore_index=True,
    )
    individual = individual.sort_values(
        ["population_size", "seed"],
    ).reset_index(drop=True)

    summary = (
        individual.groupby("population_size", as_index=False)
        .agg(
            mean_fitness=("fitness", "mean"),
            std_fitness=("fitness", "std"),
            best_fitness=("fitness", "min"),
            mean_generations=("generations", "mean"),
            mean_time_s=("elapsed_s", "mean"),
            mean_postponed_orders=("postponed_orders", "mean"),
            mean_postponed_boxes=("postponed_boxes", "mean"),
            mean_delay_days=("delay_days", "mean"),
            mean_setup_time_min=("setup_time_min", "mean"),
            mean_economic_value=("economic_value", "mean"),
            mean_capacity_utilisation=("capacity_utilisation", "mean"),
            mean_operator_minutes=("operator_minutes", "mean"),
        )
        .sort_values("population_size")
        .reset_index(drop=True)
    )
    best_population = int(
        summary.loc[summary["mean_fitness"].idxmin(), "population_size"]
    )
    summary["best_population"] = ""
    summary.loc[summary["population_size"] == best_population, "best_population"] = (
        "BEST"
    )

    configuration = pd.DataFrame(
        [
            {"parameter": "Population sizes", "value": "50, 100, 150, 200"},
            {"parameter": "Mutation rate", "value": 0.10},
            {"parameter": "Stagnation k", "value": 20},
            {"parameter": "Seeds", "value": "0, 7, 13, 42, 99"},
            {"parameter": "Elite size", "value": 5},
            {"parameter": "Tournament size", "value": 3},
            {"parameter": "Maximum generations", "value": MAX_GENERATIONS},
            {"parameter": "Productive minutes per line/day", "value": 450},
            {"parameter": "Objective version", "value": OBJECTIVE_VERSION},
            {"parameter": "Best population size", "value": best_population},
        ]
    )

    output_xlsx = Path(output_xlsx)
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        individual.to_excel(
            writer,
            sheet_name="Individual Results",
            index=False,
        )
        summary.to_excel(writer, sheet_name="Summary", index=False)
        configuration.to_excel(
            writer,
            sheet_name="Configuration",
            index=False,
        )

    print(f"Created Excel report: {output_xlsx}", flush=True)
    print(f"Best population size: {best_population}", flush=True)


def main():
    print("\n" + "=" * 50)
    print("EXPERIMENT 1 - POPULATION SIZE")
    summary_pop = run_batch(
        experiment="population_size",
        param_name="population_size",
        param_values=[50, 100, 150, 200],
        fixed_params={
            "mutation_rate": 0.10,
            "stagnation_k": 20,
            "elite_size": 5,
            "tournament_size": 3,
        },
        summary_file="results_population_size_normalised.csv",
    )
    best_pop = min(
        summary_pop,
        key=lambda row: row["mean_fitness"],
    )["population_size"]
    print(f"\nBest population size: {best_pop}")

    print("\n" + "=" * 50)
    print("EXPERIMENT 2 - MUTATION RATE")
    summary_mut = run_batch(
        experiment="mutation_rate",
        param_name="mutation_rate",
        param_values=[0.01, 0.03, 0.05, 0.08, 0.10, 0.15],
        fixed_params={
            "population_size": best_pop,
            "stagnation_k": 20,
            "elite_size": 5,
            "tournament_size": 3,
        },
        summary_file="results_mutation_rate_normalised.csv",
    )
    best_mut = min(
        summary_mut,
        key=lambda row: row["mean_fitness"],
    )["mutation_rate"]
    print(f"\nBest mutation rate: {best_mut}")

    print("\n" + "=" * 50)
    print("EXPERIMENT 3 - STOPPING CRITERION")
    run_batch(
        experiment="stagnation_k",
        param_name="stagnation_k",
        param_values=[5, 10, 20, 30, 50, MAX_GENERATIONS],
        fixed_params={
            "population_size": best_pop,
            "mutation_rate": best_mut,
            "elite_size": 5,
            "tournament_size": 3,
        },
        summary_file="results_stagnation_k_normalised.csv",
    )

    print("\nAll experiments complete.")
    print(f"Run checkpoints: {CHECKPOINT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--population-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output")
    parser.add_argument("--aggregate-population-results")
    parser.add_argument("--output-xlsx")
    args = parser.parse_args()

    if args.aggregate_population_results:
        if not args.output_xlsx:
            parser.error("--output-xlsx is required when aggregating results.")
        aggregate_population_results(
            args.aggregate_population_results,
            args.output_xlsx,
        )
    elif args.population_size is not None or args.seed is not None:
        if args.population_size is None or args.seed is None or not args.output:
            parser.error(
                "--population-size, --seed and --output must be used together."
            )
        run_population_seed(
            args.population_size,
            args.seed,
            args.output,
        )
    else:
        main()
