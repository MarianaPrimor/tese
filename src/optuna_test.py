import argparse
import csv
import json
import time
from pathlib import Path

import optuna
import pandas as pd

from generate_instance import load_real_instance
from geneticalgorithm import run_genetic_algorithm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INSTANCE_FILE = (SCRIPT_DIR / "../Inputs_June.xlsx").resolve()
N_TRIALS = 50
MAX_GENERATIONS = 200
SEEDS_FOR_GA = [42]
OBJECTIVE_VERSION = "normalised_v3_single_instance_literature_ranges"

SEED_CACHE_FIELDS = [
    "objective_version",
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
        description="Tune GA parameters with Optuna for one planning instance."
    )
    parser.add_argument(
        "--instance-file",
        type=Path,
        default=DEFAULT_INSTANCE_FILE,
        help="Excel input file to tune against.",
    )
    parser.add_argument(
        "--study-suffix",
        default="june",
        help="Suffix used to keep study/result files separate.",
    )
    parser.add_argument(
        "--operators",
        type=int,
        default=20,
        help="Number of productive operators available per day.",
    )
    parser.add_argument(
        "--non-working-dates",
        default="",
        help=(
            "Comma-separated non-working dates in YYYY-MM-DD format, "
            "in addition to weekends and the Excel holidays sheet."
        ),
    )
    return parser.parse_args()


ARGS = parse_args()
INSTANCE_FILE = ARGS.instance_file.resolve()
STUDY_SUFFIX = ARGS.study_suffix.strip().lower().replace(" ", "_")
NON_WORKING_DATES = [
    value.strip()
    for value in ARGS.non_working_dates.split(",")
    if value.strip()
]
OPERATIONAL_CONFIG = {
    "shift_start_min": 480,
    "shift_end_min": 990,
    "lunch_break_min": 30,
    "cleaning_time_min": 30,
    "standard_operators": ARGS.operators,
    "non_working_dates": NON_WORKING_DATES,
}
STUDY_NAME = f"ga_parameter_tuning_{STUDY_SUFFIX}_v1"
STORAGE_FILE = SCRIPT_DIR / f"optuna_study_{STUDY_SUFFIX}_v1.db"
STORAGE_PATH = f"sqlite:///{STORAGE_FILE.as_posix()}"
RESULTS_FILE = SCRIPT_DIR / f"optuna_results_{STUDY_SUFFIX}_v1.csv"
SEED_CACHE_FILE = SCRIPT_DIR / f"optuna_seed_checkpoint_{STUDY_SUFFIX}_v1.csv"
CONFIG_FILE = SCRIPT_DIR / f"optuna_configuration_{STUDY_SUFFIX}_v1.json"
FIGURES_DIR = SCRIPT_DIR / f"optuna_figures_{STUDY_SUFFIX}_v1"

instance = load_real_instance(
    str(INSTANCE_FILE),
    operational_config=OPERATIONAL_CONFIG,
)


def instance_signature():
    stat = INSTANCE_FILE.stat()
    return (
        f"{INSTANCE_FILE.name}:{stat.st_size}:{stat.st_mtime_ns}:"
        f"operators={ARGS.operators}:"
        f"non_working_dates={','.join(NON_WORKING_DATES)}"
    )


INSTANCE_SIGNATURE = instance_signature()


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


def cache_key(population_size, mutation_rate, stagnation_k, seed):
    return (
        OBJECTIVE_VERSION,
        INSTANCE_SIGNATURE,
        int(population_size),
        float(mutation_rate),
        int(stagnation_k),
        int(seed),
    )


def run_seed_evaluation(
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
        elite_size=5,
        tournament_size=3,
        seed=seed,
        verbose=False,
    )
    return {
        "seed": seed,
        "fitness": metrics["normalised_fitness"],
        "generations": generations,
        "elapsed_s": time.perf_counter() - start_time,
    }


def objective(trial):
    population_size = trial.suggest_int("population_size", 50, 250)
    mutation_rate = trial.suggest_float("mutation_rate", 0.01, 0.15)
    stagnation_k = trial.suggest_int("stagnation_k", 10, 60)
    fitness_by_seed = {}

    for seed in SEEDS_FOR_GA:
        key = cache_key(
            population_size,
            mutation_rate,
            stagnation_k,
            seed,
        )
        cached = seed_cache.get(key)

        if cached is not None:
            fitness = float(cached["fitness"])
            print(
                f"Trial {trial.number} seed {seed}: "
                f"using checkpoint fitness={fitness:.8f}",
                flush=True,
            )
            fitness_by_seed[seed] = fitness
            continue

        result = run_seed_evaluation(
            instance,
            population_size,
            mutation_rate,
            stagnation_k,
            seed,
        )
        fitness = result["fitness"]
        generations = result["generations"]
        elapsed = result["elapsed_s"]
        fitness_by_seed[seed] = fitness
        seed_cache[key] = {
            "objective_version": OBJECTIVE_VERSION,
            "instance_signature": INSTANCE_SIGNATURE,
            "population_size": population_size,
            "mutation_rate": mutation_rate,
            "stagnation_k": stagnation_k,
            "seed": seed,
            "fitness": f"{fitness:.12f}",
            "generations": generations,
            "elapsed_s": f"{elapsed:.3f}",
        }
        save_seed_cache()
        print(
            f"Trial {trial.number} seed {seed}: "
            f"fitness={fitness:.8f} | generations={generations} | "
            f"time={elapsed:.1f}s",
            flush=True,
        )

    fitnesses = [fitness_by_seed[seed] for seed in SEEDS_FOR_GA]

    for run_index, seed in enumerate(SEEDS_FOR_GA, start=1):
        fitness = fitness_by_seed[seed]
        trial.set_user_attr(f"seed_{seed}_fitness", fitness)
        trial.set_user_attr("completed_seed_runs", run_index)
        trial.report(
            sum(fitnesses[:run_index]) / run_index,
            step=run_index,
        )

    return sum(fitnesses) / len(fitnesses)


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
        "instance_file": str(INSTANCE_FILE),
        "instance_signature": INSTANCE_SIGNATURE,
        "study_name": STUDY_NAME,
        "storage_file": str(STORAGE_FILE),
        "n_completed_trials_target": N_TRIALS,
        "max_generations": MAX_GENERATIONS,
        "productive_minutes_per_line_day": 450,
        "operators": ARGS.operators,
        "non_working_dates": NON_WORKING_DATES,
        "seeds": SEEDS_FOR_GA,
        "population_size": [50, 250],
        "mutation_rate": [0.01, 0.15],
        "stagnation_k": [10, 60],
        "elite_size": 5,
        "tournament_size": 3,
    }
    CONFIG_FILE.write_text(
        json.dumps(configuration, indent=2),
        encoding="utf-8",
    )


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

    print(f"Instance file: {INSTANCE_FILE}")
    print(f"Study suffix: {STUDY_SUFFIX}")
    print(f"Operators: {ARGS.operators}")
    print(f"Non-working dates: {NON_WORKING_DATES}")
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
