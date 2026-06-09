import optuna
import pandas as pd
from pathlib import Path

from generate_instance import load_real_instance
from geneticalgorithm import run_genetic_algorithm


# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
INSTANCE_PATH = "../Inputs_EmpresaX.xlsx"
N_TRIALS = 60
MAX_GENERATIONS = 200
SEEDS_FOR_GA = [0, 42, 99]
STUDY_NAME = "ga_parameter_tuning"
STORAGE_FILE = SCRIPT_DIR / "optuna_study.db"
STORAGE_PATH = f"sqlite:///{STORAGE_FILE.as_posix()}"
RESULTS_FILE = SCRIPT_DIR / "optuna_results.csv"
FIGURES_DIR = SCRIPT_DIR / "optuna_figures"


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
df.to_csv(RESULTS_FILE, index=False)

print(f"\nAll trial results saved to {RESULTS_FILE}")
print(f"Study saved to {STORAGE_FILE}")


# Automatic Optuna visualizations
FIGURES_DIR.mkdir(exist_ok=True)

figures = {
    "01_optimization_history": optuna.visualization.plot_optimization_history(study),
    "02_parameter_importance": optuna.visualization.plot_param_importances(study),
    "03_contour_all_parameters": optuna.visualization.plot_contour(study),
    "04_contour_population_stagnation": optuna.visualization.plot_contour(
        study,
        params=["population_size", "stagnation_k"],
    ),
    "05_contour_mutation_stagnation": optuna.visualization.plot_contour(
        study,
        params=["mutation_rate", "stagnation_k"],
    ),
    "06_contour_population_mutation": optuna.visualization.plot_contour(
        study,
        params=["population_size", "mutation_rate"],
    ),
    "07_parallel_coordinate": optuna.visualization.plot_parallel_coordinate(study),
}

for name, fig in figures.items():
    html_path = FIGURES_DIR / f"{name}.html"
    png_path = FIGURES_DIR / f"{name}.png"

    fig.write_html(html_path)

    try:
        fig.write_image(png_path)
    except Exception as exc:
        print(
            f"Could not save {png_path.name}. "
            f"Install/repair kaleido if PNG export is needed. Error: {exc}"
        )

print(f"Optuna figures saved to {FIGURES_DIR}")
