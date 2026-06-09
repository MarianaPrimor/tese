import optuna
import pandas as pd

from generate_instance import load_real_instance
from geneticalgorithm import run_genetic_algorithm


# Configuration
INSTANCE_PATH = "../Inputs_Doceleia.xlsx"
N_TRIALS = 60
MAX_GENERATIONS = 200
SEEDS_FOR_GA = [0, 42, 99]
STUDY_NAME = "ga_parameter_tuning"
STORAGE_PATH = "sqlite:///optuna_study.db"


instance = load_real_instance(INSTANCE_PATH)


# Objective function
def objective(trial):
    population_size = trial.suggest_int("population_size", 75, 150)
    mutation_rate = trial.suggest_float("mutation_rate", 0.05, 0.15)
    stagnation_k = trial.suggest_int("stagnation_k", 10, 30)

    penalties = []

    for seed in SEEDS_FOR_GA:
        _, metrics, _ = run_genetic_algorithm(
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

        penalties.append(metrics["total_penalty"])

    return sum(penalties) / len(penalties)


# Run the study
optuna.logging.set_verbosity(optuna.logging.INFO)

study = optuna.create_study(
    direction="minimize",
    study_name=STUDY_NAME,
    storage=STORAGE_PATH,
    load_if_exists=True,
    sampler=optuna.samplers.TPESampler(seed=42),
)

study.optimize(objective, n_trials=N_TRIALS)


# Results
print("\n" + "=" * 50)
print("OPTUNA RESULTS")
print("=" * 50)
print(f"Best trial:     #{study.best_trial.number}")
print(f"Best fitness:   {study.best_value:.0f}")
print("Best params:")

for key, value in study.best_params.items():
    print(f"  {key}: {value}")

df = study.trials_dataframe()
df.to_csv("optuna_results.csv", index=False)

print("\nAll trial results saved to optuna_results.csv")
print("Study saved to optuna_study.db")
