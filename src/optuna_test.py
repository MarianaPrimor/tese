import csv
import json
import time
from pathlib import Path

import optuna
import pandas as pd

from generate_instance import load_real_instance
from geneticalgorithm import run_genetic_algorithm


SCRIPT_DIR = Path(__file__).resolve().parent
INSTANCE_FILE = (SCRIPT_DIR / "../Inputs_EmpresaX.xlsx").resolve()
N_TRIALS = 60
MAX_GENERATIONS = 200
SEEDS_FOR_GA = [0, 42, 99]
STUDY_NAME = "ga_parameter_tuning_normalised_v2"
STORAGE_FILE = SCRIPT_DIR / "optuna_study_normalised.db"
STORAGE_PATH = f"sqlite:///{STORAGE_FILE.as_posix()}"
RESULTS_FILE = SCRIPT_DIR / "optuna_results_normalised.csv"
SEED_CACHE_FILE = SCRIPT_DIR / "optuna_seed_checkpoint_normalised.csv"
CONFIG_FILE = SCRIPT_DIR / "optuna_configuration_normalised.json"
FIGURES_DIR = SCRIPT_DIR / "optuna_figures_normalised"
OBJECTIVE_VERSION = "normalised_v2_l1_l2_finishing"

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

instance = load_real_instance(str(INSTANCE_FILE))


def instance_signature():
    stat = INSTANCE_FILE.stat()
    return f"{INSTANCE_FILE.name}:{stat.st_size}:{stat.st_mtime_ns}"


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


def objective(trial):
    population_size = trial.suggest_int("population_size", 150, 200)
    mutation_rate = trial.suggest_float("mutation_rate", 0.05, 0.10)
    stagnation_k = trial.suggest_int("stagnation_k", 20, 60)
    fitnesses = []

    for run_index, seed in enumerate(SEEDS_FOR_GA, start=1):
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
        else:
            print(
                f"Trial {trial.number} seed {seed} "
                f"({run_index}/{len(SEEDS_FOR_GA)}): running",
                flush=True,
            )
            start_time = time.perf_counter()
            _, metrics, generations = run_genetic_algorithm(
                instance,
                population_size=population_size,
                mutation_rate=mutation_rate,
                stagnation_k=stagnation_k,
                generations=MAX_GENERATIONS,
                elite_size=5,
                tournament_size=3,
                seed=seed,
                verbose=False,
            )
            elapsed = time.perf_counter() - start_time
            fitness = metrics["normalised_fitness"]
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

        fitnesses.append(fitness)
        trial.set_user_attr(f"seed_{seed}_fitness", fitness)
        trial.set_user_attr("completed_seed_runs", len(fitnesses))
        trial.report(sum(fitnesses) / len(fitnesses), step=run_index)

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
        "seeds": SEEDS_FOR_GA,
        "population_size": [150, 200],
        "mutation_rate": [0.05, 0.10],
        "stagnation_k": [20, 60],
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

    try:
        from generate_optuna_figures import main as generate_figures

        generate_figures()
    except Exception as exc:
        print(f"Automatic figure generation failed: {exc}")
        print("The study is safe. Regenerate figures later with:")
        print("  python generate_optuna_figures.py")


if __name__ == "__main__":
    main()
