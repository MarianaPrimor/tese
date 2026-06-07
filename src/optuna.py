import optuna
import time
from generate_instance import load_real_instance
from geneticalgorithm import run_genetic_algorithm

# ── Configuration ─────────────────────────────────────────────
INSTANCE_PATH  = "../Inputs_Doceleia.xlsx"
N_TRIALS       = 50
MAX_GENERATIONS = 200
SEED_FOR_GA    = 42   # fixed seed so each trial is reproducible

instance = load_real_instance(INSTANCE_PATH)

# ── Objective function ────────────────────────────────────────
def objective(trial):

    # Optuna suggests values within your defined ranges
    population_size = trial.suggest_int("population_size", 75, 150)
    mutation_rate   = trial.suggest_float("mutation_rate", 0.05, 0.15)
    stagnation_k    = trial.suggest_int("stagnation_k", 10, 30)

    # Run the GA with these parameters
    _, metrics, _ = run_genetic_algorithm(
        instance,
        population_size  = population_size,
        mutation_rate    = mutation_rate,
        stagnation_k     = stagnation_k,
        generations      = MAX_GENERATIONS,
        elite_size       = 5,
        tournament_size  = 3,
        seed             = SEED_FOR_GA,
        verbose          = False,
    )

    return metrics["total_penalty"]

# ── Run the study ──────────────────────────────────────────────
optuna.logging.set_verbosity(optuna.logging.INFO)

study = optuna.create_study(
    direction="minimize",
    study_name="ga_parameter_tuning",
    sampler=optuna.samplers.TPESampler(seed=42),
)

study.optimize(objective, n_trials=N_TRIALS)

# ── Results ────────────────────────────────────────────────────
print("\n" + "="*50)
print("OPTUNA RESULTS")
print("="*50)
print(f"Best trial:     #{study.best_trial.number}")
print(f"Best fitness:   {study.best_value:.0f}")
print(f"Best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# Save all trials to CSV
import pandas as pd
df = study.trials_dataframe()
df.to_csv("optuna_results.csv", index=False)
print("\nAll trial results saved to optuna_results.csv")
