import argparse
from pathlib import Path

import optuna
import pandas as pd
import plotly.express as px


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Optuna diagnostic figures for one study."
    )
    parser.add_argument(
        "--study-suffix",
        default="june",
        help="Suffix used by optuna_test.py, e.g. june, july, august.",
    )
    return parser.parse_args()


def paths_for_suffix(study_suffix):
    suffix = study_suffix.strip().lower().replace(" ", "_")
    study_name = f"ga_parameter_tuning_{suffix}_v1"
    storage_file = SCRIPT_DIR / f"optuna_study_{suffix}_v1.db"
    storage_path = f"sqlite:///{storage_file.as_posix()}"
    figures_dir = SCRIPT_DIR / f"optuna_figures_{suffix}_v1"
    return suffix, study_name, storage_file, storage_path, figures_dir


def save_figure(figures_dir, name, factory):
    try:
        figure = factory()
        html_path = figures_dir / f"{name}.html"
        figure.write_html(html_path)
        print(f"Saved HTML: {html_path.name}")

        try:
            png_path = figures_dir / f"{name}.png"
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


def correlation_figure(study):
    records = []

    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        row = {"fitness": trial.value}
        row.update(trial.params)
        records.append(row)

    if len(records) < 2:
        raise RuntimeError("At least two completed trials are required.")

    df = pd.DataFrame(records)
    columns = [
        column
        for column in [
            "fitness",
            "population_size",
            "mutation_rate",
            "stagnation_k",
        ]
        if column in df.columns
    ]
    corr = df[columns].corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation matrix between Optuna parameters and fitness",
    )
    fig.update_layout(width=850, height=700)
    return fig


def main():
    args = parse_args()
    suffix, study_name, storage_file, storage_path, figures_dir = paths_for_suffix(
        args.study_suffix
    )

    if not storage_file.exists():
        raise FileNotFoundError(
            f"Study database not found: {storage_file}"
        )

    study = optuna.load_study(
        study_name=study_name,
        storage=storage_path,
    )
    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]

    if not completed:
        raise RuntimeError("The study has no completed trials yet.")

    figures_dir.mkdir(exist_ok=True)
    figures = [
        (
            "01_optimization_history",
            lambda: optuna.visualization.plot_optimization_history(study),
        ),
        (
            "02_anova_parameter_importance",
            lambda: parameter_importance_figure(study),
        ),
        (
            "03_parameter_correlation_matrix",
            lambda: correlation_figure(study),
        ),
        (
            "04_contour_all_parameters",
            lambda: optuna.visualization.plot_contour(study),
        ),
        (
            "05_contour_population_stagnation",
            lambda: optuna.visualization.plot_contour(
                study,
                params=["population_size", "stagnation_k"],
            ),
        ),
        (
            "06_contour_mutation_stagnation",
            lambda: optuna.visualization.plot_contour(
                study,
                params=["mutation_rate", "stagnation_k"],
            ),
        ),
        (
            "07_contour_population_mutation",
            lambda: optuna.visualization.plot_contour(
                study,
                params=["population_size", "mutation_rate"],
            ),
        ),
        (
            "08_parallel_coordinate",
            lambda: optuna.visualization.plot_parallel_coordinate(study),
        ),
        (
            "09_slice",
            lambda: optuna.visualization.plot_slice(study),
        ),
    ]

    successes = sum(
        save_figure(figures_dir, name, factory)
        for name, factory in figures
    )
    print(
        f"\nGenerated {successes}/{len(figures)} figure sets from "
        f"{len(completed)} completed trials for {suffix}."
    )
    print(f"Figures directory: {figures_dir}")


if __name__ == "__main__":
    main()
