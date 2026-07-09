import argparse
import csv
import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import optuna
import pandas as pd

from generate_instance import load_real_instance
from geneticalgorithm import run_genetic_algorithm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INSTANCE_FILES = {
    "june": (SCRIPT_DIR / "../Inputs_June.xlsx").resolve(),
    "july": (SCRIPT_DIR / "../Inputs_July.xlsx").resolve(),
    "august": (SCRIPT_DIR / "../Inputs_August.xlsx").resolve(),
}
N_TRIALS = 30
MAX_GENERATIONS = 200
SEEDS_FOR_GA = [42, 43, 44]
OBJECTIVE_VERSION = "normalised_v5_new_algorithm_3seeds_30trials"

SEED_CACHE_FIELDS = [
    "objective_version",
    "instance_name",
    "instance_signature",
    "population_size",
    "mutation_rate",
    "stagnation_k",
    "seed",
    "fitness",
    "generations",
    "elapsed_s",
]



def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune GA parameters with Optuna across multiple planning instances."
    )
    parser.add_argument(
        "--study-suffix",
        default="multi_instance",
        help="Suffix used to keep study/result files separate.",
    )
    parser.add_argument(
        "--operators",
        type=int,
        default=20,
        help="Number of productive operators available per day.",
    )
    parser.add_argument("--june-instance-file", type=Path, default=DEFAULT_INSTANCE_FILES["june"])
    parser.add_argument("--july-instance-file", type=Path, default=DEFAULT_INSTANCE_FILES["july"])
    parser.add_argument("--august-instance-file", type=Path, default=DEFAULT_INSTANCE_FILES["august"])
    parser.add_argument("--june-non-working-dates", default="")
    parser.add_argument("--july-non-working-dates", default="")
    parser.add_argument("--august-non-working-dates", default="")
    # Backwards-compatible single-instance arguments. If provided, the script runs
    # exactly one instance, which is useful for older workflows or quick checks.
    parser.add_argument("--instance-file", type=Path, default=None)
    parser.add_argument("--non-working-dates", default=None)
    return parser.parse_args()


ARGS = parse_args()
STUDY_SUFFIX = ARGS.study_suffix.strip().lower().replace(" ", "_")
STUDY_NAME = f"ga_parameter_tuning_{STUDY_SUFFIX}_v1"
STORAGE_FILE = SCRIPT_DIR / f"optuna_study_{STUDY_SUFFIX}_v1.db"
STORAGE_PATH = f"sqlite:///{STORAGE_FILE.as_posix()}"
RESULTS_FILE = SCRIPT_DIR / f"optuna_results_{STUDY_SUFFIX}_v1.csv"
SEED_CACHE_FILE = SCRIPT_DIR / f"optuna_seed_checkpoint_{STUDY_SUFFIX}_v1.csv"
CONFIG_FILE = SCRIPT_DIR / f"optuna_configuration_{STUDY_SUFFIX}_v1.json"
FIGURES_DIR = SCRIPT_DIR / f"optuna_figures_{STUDY_SUFFIX}_v1"


def split_dates(value):
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


if ARGS.instance_file is not None:
    INSTANCE_FILES = {STUDY_SUFFIX: ARGS.instance_file.resolve()}
    NON_WORKING_DATES_BY_INSTANCE = {
        STUDY_SUFFIX: split_dates(ARGS.non_working_dates or "")
    }
else:
    INSTANCE_FILES = {
        "june": ARGS.june_instance_file.resolve(),
        "july": ARGS.july_instance_file.resolve(),
        "august": ARGS.august_instance_file.resolve(),
    }
    NON_WORKING_DATES_BY_INSTANCE = {
        "june": split_dates(ARGS.june_non_working_dates),
        "july": split_dates(ARGS.july_non_working_dates),
        "august": split_dates(ARGS.august_non_working_dates),
    }


def build_operational_config(instance_name):
    return {
        "shift_start_min": 480,
        "shift_end_min": 990,
        "lunch_break_min": 30,
        "cleaning_time_min": 30,
        "standard_operators": ARGS.operators,
        "non_working_dates": NON_WORKING_DATES_BY_INSTANCE.get(instance_name, []),
    }


def load_instances():
    loaded = {}
    for name, path in INSTANCE_FILES.items():
        loaded[name] = load_real_instance(
            str(path),
            operational_config=build_operational_config(name),
        )
    return loaded


instances = load_instances()


def instance_signature(instance_name):
    path = INSTANCE_FILES[instance_name]
    stat = path.stat()
    return (
        f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}:"
        f"operators={ARGS.operators}:"
        f"non_working_dates={','.join(NON_WORKING_DATES_BY_INSTANCE.get(instance_name, []))}"
    )


INSTANCE_SIGNATURES = {
    name: instance_signature(name)
    for name in INSTANCE_FILES
}


def atomic_write_dataframe(df, path):
    temp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp_path, index=False)
    temp_path.replace(path)


def load_seed_cache():
    if not SEED_CACHE_FILE.exists():
        return {}

    cache = {}
    with SEED_CACHE_FILE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["objective_version"],
                row.get("instance_name", STUDY_SUFFIX),
                row["instance_signature"],
                int(row["population_size"]),
                float(row["mutation_rate"]),
                int(row["stagnation_k"]),
                int(row["seed"]),
            )
            cache[key] = row
    return cache


seed_cache = load_seed_cache()


def save_seed_cache():
    temp_path = SEED_CACHE_FILE.with_suffix(".csv.tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEED_CACHE_FIELDS)
        writer.writeheader()
        writer.writerows(seed_cache.values())
    temp_path.replace(SEED_CACHE_FILE)


def cache_key(instance_name, population_size, mutation_rate, stagnation_k, seed):
    return (
        OBJECTIVE_VERSION,
        instance_name,
        INSTANCE_SIGNATURES[instance_name],
        int(population_size),
        float(mutation_rate),
        int(stagnation_k),
        int(seed),
    )


def run_seed_evaluation(
    instance_name,
    instance_data,
    population_size,
    mutation_rate,
    stagnation_k,
    seed,
):
    start_time = time.perf_counter()
    _, metrics, generations = run_genetic_algorithm(
        instance_data,
        population_size=population_size,
        mutation_rate=mutation_rate,
        stagnation_k=stagnation_k,
        generations=MAX_GENERATIONS,
        elite_size=None,
        tournament_size=3,
        seed=seed,
        verbose=False,
    )
    return {
        "instance_name": instance_name,
        "seed": seed,
        "fitness": metrics["normalised_fitness"],
        "generations": generations,
        "elapsed_s": time.perf_counter() - start_time,
    }


def objective(trial):
    population_size = trial.suggest_int("population_size", 50, 250)
    mutation_rate = trial.suggest_float("mutation_rate", 0.01, 0.15)
    stagnation_k = trial.suggest_int("stagnation_k", 10, 60)

    run_results = []
    pending_runs = []
    total_runs = len(instances) * len(SEEDS_FOR_GA)

    for instance_name, instance_data in instances.items():
        for seed in SEEDS_FOR_GA:
            key = cache_key(
                instance_name,
                population_size,
                mutation_rate,
                stagnation_k,
                seed,
            )
            cached = seed_cache.get(key)

            if cached is not None:
                result = {
                    "instance_name": instance_name,
                    "seed": seed,
                    "fitness": float(cached["fitness"]),
                    "generations": int(cached["generations"]),
                    "elapsed_s": float(cached["elapsed_s"]),
                }
                run_results.append(result)
                print(
                    f"Trial {trial.number} {instance_name} seed {seed}: "
                    f"using checkpoint fitness={result['fitness']:.8f}",
                    flush=True,
                )
            else:
                pending_runs.append((instance_name, instance_data, seed, key))

    completed_runs = len(run_results)
    if completed_runs:
        running_mean = sum(item["fitness"] for item in run_results) / completed_runs
        trial.set_user_attr("completed_seed_runs", completed_runs)
        trial.report(running_mean, step=completed_runs)

    if pending_runs:
        max_workers = min(total_runs, len(pending_runs))
        print(
            f"Trial {trial.number}: running {len(pending_runs)} seed/instance "
            f"evaluations in parallel with {max_workers} workers.",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_run = {
                executor.submit(
                    run_seed_evaluation,
                    instance_name,
                    instance_data,
                    population_size,
                    mutation_rate,
                    stagnation_k,
                    seed,
                ): (instance_name, seed, key)
                for instance_name, instance_data, seed, key in pending_runs
            }

            for future in as_completed(future_to_run):
                instance_name, seed, key = future_to_run[future]
                result = future.result()
                fitness = result["fitness"]
                generations = result["generations"]
                elapsed = result["elapsed_s"]
                seed_cache[key] = {
                    "objective_version": OBJECTIVE_VERSION,
                    "instance_name": instance_name,
                    "instance_signature": INSTANCE_SIGNATURES[instance_name],
                    "population_size": population_size,
                    "mutation_rate": mutation_rate,
                    "stagnation_k": stagnation_k,
                    "seed": seed,
                    "fitness": f"{fitness:.12f}",
                    "generations": generations,
                    "elapsed_s": f"{elapsed:.3f}",
                }
                run_results.append(result)
                completed_runs += 1
                save_seed_cache()
                print(
                    f"Trial {trial.number} {instance_name} seed {seed}: "
                    f"fitness={fitness:.8f} | generations={generations} | "
                    f"time={elapsed:.1f}s",
                    flush=True,
                )

                running_mean = sum(item["fitness"] for item in run_results) / len(run_results)
                trial.set_user_attr("completed_seed_runs", completed_runs)
                trial.report(running_mean, step=completed_runs)

    fitnesses = [item["fitness"] for item in run_results]
    mean_fitness = sum(fitnesses) / len(fitnesses)
    trial.set_user_attr("mean_fitness", mean_fitness)
    trial.set_user_attr("std_fitness", statistics.pstdev(fitnesses) if len(fitnesses) > 1 else 0.0)
    trial.set_user_attr("worst_fitness", max(fitnesses))
    trial.set_user_attr("best_fitness", min(fitnesses))

    for instance_name in instances:
        values = [
            item["fitness"]
            for item in run_results
            if item["instance_name"] == instance_name
        ]
        trial.set_user_attr(f"mean_{instance_name}_fitness", sum(values) / len(values))

    for item in run_results:
        trial.set_user_attr(
            f"{item['instance_name']}_seed_{item['seed']}_fitness",
            item["fitness"],
        )

    return mean_fitness


def export_study(study, trial=None):
    atomic_write_dataframe(study.trials_dataframe(), RESULTS_FILE)
    completed = sum(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in study.trials
    )
    print(
        f"Checkpoint saved: {completed} completed trials -> {RESULTS_FILE.name}",
        flush=True,
    )


def recover_interrupted_trials(study):
    interrupted = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.RUNNING
    ]

    for trial in interrupted:
        params = dict(trial.params)
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        if params:
            study.enqueue_trial(
                params,
                user_attrs={"recovered_from_trial": trial.number},
            )
        print(
            f"Recovered interrupted trial #{trial.number}; "
            "its parameter combination was queued again.",
            flush=True,
        )


def save_configuration():
    configuration = {
        "objective_version": OBJECTIVE_VERSION,
        "instance_files": {name: str(path) for name, path in INSTANCE_FILES.items()},
        "instance_signatures": INSTANCE_SIGNATURES,
        "study_name": STUDY_NAME,
        "storage_file": str(STORAGE_FILE),
        "n_completed_trials_target": N_TRIALS,
        "runs_per_trial": len(instances) * len(SEEDS_FOR_GA),
        "max_generations": MAX_GENERATIONS,
        "productive_minutes_per_line_day": 450,
        "operators": ARGS.operators,
        "non_working_dates_by_instance": NON_WORKING_DATES_BY_INSTANCE,
        "seeds": SEEDS_FOR_GA,
        "population_size": [50, 250],
        "mutation_rate": [0.01, 0.15],
        "stagnation_k": [10, 60],
        "elite_size": "10% of population",
        "tournament_size": 3,
        "objective_aggregation": "mean fitness across instances and seeds",
    }
    CONFIG_FILE.write_text(json.dumps(configuration, indent=2), encoding="utf-8")


def main():
    optuna.logging.set_verbosity(optuna.logging.INFO)
    save_configuration()
    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        storage=STORAGE_PATH,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    recover_interrupted_trials(study)
    export_study(study)

    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    remaining_trials = max(0, N_TRIALS - len(completed_trials))

    print(f"Study suffix: {STUDY_SUFFIX}")
    print(f"Instances: {INSTANCE_FILES}")
    print(f"Seeds: {SEEDS_FOR_GA}")
    print(f"Runs per trial: {len(instances) * len(SEEDS_FOR_GA)}")
    print(f"Operators: {ARGS.operators}")
    print(f"Non-working dates by instance: {NON_WORKING_DATES_BY_INSTANCE}")
    print(f"Completed trials already stored: {len(completed_trials)}")
    print(f"Remaining completed trials required: {remaining_trials}")

    if remaining_trials > 0:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            callbacks=[export_study],
            gc_after_trial=True,
            catch=(Exception,),
        )
    else:
        print("Target number of completed trials already reached.")

    export_study(study)
    print("\n" + "=" * 50)
    print("OPTUNA RESULTS")
    print("=" * 50)
    print(f"Best trial:   #{study.best_trial.number}")
    print(f"Best fitness: {study.best_value:.8f}")
    print("Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"\nStudy saved to {STORAGE_FILE}")
    print(f"Seed checkpoints saved to {SEED_CACHE_FILE}")


if __name__ == "__main__":
    main()