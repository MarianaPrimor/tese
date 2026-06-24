from pathlib import Path

import optuna


SCRIPT_DIR = Path(__file__).resolve().parent
STUDY_NAME = "ga_parameter_tuning_normalised_v3_single_seed"
STORAGE_FILE = SCRIPT_DIR / "optuna_study_normalised_v3_single_seed.db"
STORAGE_PATH = f"sqlite:///{STORAGE_FILE.as_posix()}"
FIGURES_DIR = SCRIPT_DIR / "optuna_figures_normalised_v3_single_seed"


def save_figure(name, factory):
    try:
        figure = factory()
        html_path = FIGURES_DIR / f"{name}.html"
        figure.write_html(html_path)
        print(f"Saved HTML: {html_path.name}")

        try:
            png_path = FIGURES_DIR / f"{name}.png"
            figure.write_image(png_path)
            print(f"Saved PNG:  {png_path.name}")
        except Exception as exc:
            print(f"PNG skipped for {name}: {exc}")

        return True
    except Exception as exc:
        print(f"Figure failed ({name}): {exc}")
        return False


def parameter_importance_figure(study):
    try:
        evaluator = optuna.importance.FanovaImportanceEvaluator(seed=42)
        return optuna.visualization.plot_param_importances(
            study,
            evaluator=evaluator,
        )
    except Exception as exc:
        print(f"fANOVA unavailable, using PED-ANOVA: {exc}")
        evaluator = optuna.importance.PedAnovaImportanceEvaluator()
        return optuna.visualization.plot_param_importances(
            study,
            evaluator=evaluator,
        )


def main():
    if not STORAGE_FILE.exists():
        raise FileNotFoundError(
            f"Study database not found: {STORAGE_FILE}"
        )

    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=STORAGE_PATH,
    )
    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]

    if not completed:
        raise RuntimeError("The study has no completed trials yet.")

    FIGURES_DIR.mkdir(exist_ok=True)
    figures = [
        (
            "01_optimization_history",
            lambda: optuna.visualization.plot_optimization_history(study),
        ),
        (
            "02_parameter_importance",
            lambda: parameter_importance_figure(study),
        ),
        (
            "03_contour_all_parameters",
            lambda: optuna.visualization.plot_contour(study),
        ),
        (
            "04_contour_population_stagnation",
            lambda: optuna.visualization.plot_contour(
                study,
                params=["population_size", "stagnation_k"],
            ),
        ),
        (
            "05_contour_mutation_stagnation",
            lambda: optuna.visualization.plot_contour(
                study,
                params=["mutation_rate", "stagnation_k"],
            ),
        ),
        (
            "06_contour_population_mutation",
            lambda: optuna.visualization.plot_contour(
                study,
                params=["population_size", "mutation_rate"],
            ),
        ),
        (
            "07_parallel_coordinate",
            lambda: optuna.visualization.plot_parallel_coordinate(study),
        ),
        (
            "08_slice",
            lambda: optuna.visualization.plot_slice(study),
        ),
    ]

    successes = sum(save_figure(name, factory) for name, factory in figures)
    print(
        f"\nGenerated {successes}/{len(figures)} figure sets from "
        f"{len(completed)} completed trials."
    )
    print(f"Figures directory: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
