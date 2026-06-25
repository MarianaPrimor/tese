import argparse
import copy
import io
import json
import math
import os
import time
from contextlib import redirect_stdout
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from evaluator import DEFAULT_NORMALISED_WEIGHTS
from generate_instance import load_real_instance
from geneticalgorithm import run_genetic_algorithm


SEEDS = [0, 42, 99]
DEMAND_FACTORS = [0.6, 0.7, 0.8, 0.9, 1.0]
OPERATOR_COUNTS = list(range(18, 31))
WEIGHT_VALUES = [round(i / 10, 1) for i in range(11)]

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DEFAULT_EXCEL_PATH = PROJECT_DIR / "Inputs_EmpresaX.xlsx"
OUTPUT_DIR = BASE_DIR / "outputs" / "scenario_analysis"
RESULTS_DIR = OUTPUT_DIR / "results"
PLOTS_DIR = OUTPUT_DIR / "plots"

GA_PARAMETERS = {
    "population_size": 162,
    "generations": 200,
    "mutation_rate": 0.07463622448274694,
    "elite_size": 5,
    "tournament_size": 3,
    "stagnation_k": 24,
}

OPTUNA_BEST_TRIAL = 32
OPTUNA_BEST_FITNESS = -0.35901551658187786

METRIC_LABELS = {
    "postponed_orders": "Postponed orders",
    "postponed_boxes": "Postponed boxes",
    "euros_produced": "Economic value produced",
    "fulfillment_rate": "Fulfilment rate",
    "fitness": "Normalised fitness",
    "kilos": "Kilograms produced",
    "total_setup_time": "Setup time",
    "operators_used": "Operator-minutes used",
}

WEIGHT_LABELS = {
    "postponement": "Postponed orders",
    "economic_value": "Economic value",
    "delay": "Delivery delay",
    "setup": "Setup time",
    "capacity_utilisation": "Capacity utilisation",
    "operator_utilisation": "Operator utilisation",
}


def ensure_output_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def configure_output_dir(output_dir):
    global OUTPUT_DIR, RESULTS_DIR, PLOTS_DIR

    OUTPUT_DIR = Path(output_dir).resolve()
    RESULTS_DIR = OUTPUT_DIR / "results"
    PLOTS_DIR = OUTPUT_DIR / "plots"


OPERATIONAL_CONFIG = {}


def load_base_instance(excel_path):
    operational_config = dict(OPERATIONAL_CONFIG)

    if not operational_config:
        operational_config = json.loads(
            os.environ.get("SCENARIO_OPERATIONAL_CONFIG", "{}")
        )

    return load_real_instance(
        excel_path=str(excel_path),
        operational_config=operational_config,
    )


def parse_hhmm(value):
    hours, minutes = str(value).split(":", maxsplit=1)
    return int(hours) * 60 + int(minutes)


def configure_operational_inputs(args):
    global OPERATIONAL_CONFIG

    OPERATIONAL_CONFIG = {
        "planning_start_date": args.planning_start,
        "planning_end_date": args.planning_end,
        "standard_operators": args.operators,
        "shifts": args.shifts,
        "shift_start_min": parse_hhmm(args.shift_start),
        "shift_end_min": parse_hhmm(args.shift_end),
        "lunch_break_min": args.lunch_minutes,
        "cleaning_time_min": args.cleaning_minutes,
        "non_working_dates": [
            value.strip()
            for value in args.non_working_dates.split(",")
            if value.strip()
        ],
    }
    os.environ["SCENARIO_OPERATIONAL_CONFIG"] = json.dumps(OPERATIONAL_CONFIG)


def total_demand_boxes(instance):
    return sum(order.get("master_boxes", 0) or 0 for order in instance.get("demand", []))


def scale_demand(instance, factor):
    scenario = copy.deepcopy(instance)

    for order in scenario.get("demand", []):
        original_quantity = order.get("master_boxes", 0) or 0

        if original_quantity <= 0:
            order["master_boxes"] = 0
        else:
            order["master_boxes"] = max(1, int(round(original_quantity * factor)))

    scenario.setdefault("_meta", {})
    scenario["_meta"]["scenario_demand_factor"] = factor
    return scenario


def set_uniform_operators(instance, total_operators):
    scenario = copy.deepcopy(instance)
    total_operators = int(total_operators)
    scenario["standard_operators"] = total_operators
    scenario["standard_operators_by_day"] = {
        day: total_operators
        for day in range(1, scenario.get("n_days", 0) + 1)
    }
    scenario.setdefault("_meta", {})
    scenario["_meta"]["scenario_total_operators"] = total_operators
    return scenario


def normalise_weights(weights):
    total = sum(max(0, value) for value in weights.values())

    if total <= 0:
        return dict(DEFAULT_NORMALISED_WEIGHTS)

    return {
        key: max(0, value) / total
        for key, value in weights.items()
    }


def make_ofat_weights(weight_name, weight_value):
    base_weights = dict(DEFAULT_NORMALISED_WEIGHTS)
    other_names = [name for name in base_weights if name != weight_name]
    remaining_weight = max(0.0, 1.0 - weight_value)
    other_total = sum(base_weights[name] for name in other_names)

    weights = {weight_name: weight_value}

    if other_total <= 0:
        equal_share = remaining_weight / len(other_names)
        for name in other_names:
            weights[name] = equal_share
    else:
        for name in other_names:
            weights[name] = remaining_weight * base_weights[name] / other_total

    return normalise_weights(weights)


def run_ga(instance, seed, objective_weights=None):
    solution, metrics, actual_generations = run_genetic_algorithm(
        instance,
        seed=seed,
        verbose=False,
        objective_weights=objective_weights,
        **GA_PARAMETERS,
    )
    metrics["actual_generations"] = actual_generations
    return solution, metrics


def refs_by_id(instance):
    return {
        str(ref.get("id")).strip(): ref
        for ref in instance.get("refs", [])
    }


def postponed_by_abc(solution, instance):
    refs = refs_by_id(instance)
    counts = {"A": 0, "B": 0, "C": 0}

    for item in solution:
        if not item.get("postponed"):
            continue

        ref = refs.get(str(item.get("ref_id")).strip(), {})
        abc_class = str(
            item.get("abc_class")
            or ref.get("abc_class")
            or "C"
        ).strip().upper()

        if abc_class not in counts:
            abc_class = "C"

        counts[abc_class] += 1

    return counts


def extract_common_metrics(metrics, instance):
    demand_boxes = max(1, total_demand_boxes(instance))
    postponed_boxes = metrics.get("postponed_boxes", 0) or 0

    return {
        "postponed_orders": metrics.get("postponed_orders", 0),
        "postponed_boxes": postponed_boxes,
        "euros_produced": metrics.get("scheduled_economic_value", 0),
        "fulfillment_rate": 1 - postponed_boxes / demand_boxes,
        "fitness": metrics.get("normalised_fitness", math.nan),
        "kilos": metrics.get("scheduled_kg", 0),
        "total_setup_time": metrics.get("setup_total_min", 0),
        "operators_used": metrics.get("operator_usage_minutes", 0),
        "delay_days_total": metrics.get("delay_days_total", 0),
        "capacity_utilisation": metrics.get(
            "capacity_utilisation_ratio",
            0,
        ),
        "computation_time_sec": metrics.get("computation_time_sec", 0),
        "actual_generations": metrics.get("actual_generations", 0),
    }


def run_axis_1_case(args):
    excel_path, demand_factor, seed = args

    with redirect_stdout(io.StringIO()):
        instance = scale_demand(load_base_instance(excel_path), demand_factor)
        solution, metrics = run_ga(instance, seed)

    row = {
        "demand_factor": demand_factor,
        "seed": seed,
    }
    row.update(extract_common_metrics(metrics, instance))
    abc_counts = postponed_by_abc(solution, instance)
    row.update({
        "postponed_A": abc_counts["A"],
        "postponed_B": abc_counts["B"],
        "postponed_C": abc_counts["C"],
    })
    return row


def run_axis_2_case(args):
    excel_path, total_operators, seed = args

    with redirect_stdout(io.StringIO()):
        instance = set_uniform_operators(load_base_instance(excel_path), total_operators)
        _solution, metrics = run_ga(instance, seed)

    row = {
        "total_operators": total_operators,
        "seed": seed,
    }
    row.update(extract_common_metrics(metrics, instance))
    return row


def run_axis_3_case(args):
    excel_path, weight_name, weight_value, seed = args

    with redirect_stdout(io.StringIO()):
        instance = load_base_instance(excel_path)

        if weight_name == "baseline":
            objective_weights = dict(DEFAULT_NORMALISED_WEIGHTS)
        else:
            objective_weights = make_ofat_weights(weight_name, weight_value)

        _solution, metrics = run_ga(instance, seed, objective_weights=objective_weights)

    row = {
        "weight_name": weight_name,
        "weight_value": weight_value,
        "seed": seed,
    }

    for name, value in objective_weights.items():
        row[f"used_weight_{name}"] = value

    row.update(extract_common_metrics(metrics, instance))
    return row


def execute_parallel(jobs, worker, max_workers=None):
    rows = []
    max_workers = max_workers or max(1, min(os.cpu_count() or 1, len(jobs)))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, job) for job in jobs]

        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            print(f"Completed {index}/{len(jobs)} runs", flush=True)

    return rows


def save_axis_csv(axis, rows):
    path = RESULTS_DIR / f"axis_{axis}.csv"
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def save_summary_csv(axis, df, group_cols, metrics):
    path = RESULTS_DIR / f"axis_{axis}_summary.csv"
    summary = df.groupby(group_cols)[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    summary.to_csv(path, index=False)
    return path


def aggregate_by(df, x_col, metrics):
    grouped = df.groupby(x_col)[metrics].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in grouped.columns
    ]
    return grouped


def save_line_plot_with_std(
    df,
    x_col,
    y_col,
    title,
    xlabel,
    ylabel,
    output_path,
    vertical_line=None,
    vertical_label=None,
):
    summary = aggregate_by(df, x_col, [y_col])
    x = summary[x_col]
    mean = summary[f"{y_col}_mean"]
    std = summary[f"{y_col}_std"].fillna(0)

    plt.figure(figsize=(9, 5))
    plt.plot(x, mean, marker="o", color="#123C7C", linewidth=2)
    plt.fill_between(
        x,
        mean - std,
        mean + std,
        color="#123C7C",
        alpha=0.18,
        linewidth=0,
    )

    if vertical_line is not None:
        plt.axvline(vertical_line, color="#B00032", linestyle="--", linewidth=1.5)
        if vertical_label:
            plt.text(
                vertical_line,
                plt.ylim()[1],
                vertical_label,
                color="#B00032",
                ha="right",
                va="top",
                rotation=90,
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def first_positive_x(df, x_col, y_col):
    summary = df.groupby(x_col)[y_col].mean().reset_index().sort_values(x_col)
    positive = summary[summary[y_col] > 0]

    if positive.empty:
        return None

    return positive.iloc[0][x_col]


def plot_axis_1_break_even(df):
    path = PLOTS_DIR / "axis_1_break_even_postponement_threshold.png"
    summary = aggregate_by(df, "demand_factor", ["postponed_orders"])
    x = summary["demand_factor"]
    mean = summary["postponed_orders_mean"]
    std = summary["postponed_orders_std"].fillna(0)
    threshold = first_positive_x(df, "demand_factor", "postponed_orders")

    plt.figure(figsize=(9, 5))
    plt.plot(x, mean, marker="o", color="#123C7C", linewidth=2)
    plt.fill_between(
        x,
        mean - std,
        mean + std,
        color="#123C7C",
        alpha=0.18,
        linewidth=0,
    )

    if threshold is not None:
        plt.axvline(threshold, color="#B00032", linestyle="--", linewidth=1.5)
        plt.text(
            threshold,
            plt.ylim()[1],
            f"First postponement: {threshold:.1f}",
            color="#B00032",
            ha="right",
            va="top",
            rotation=90,
        )

    plt.axvline(1.0, color="#555555", linestyle=":", linewidth=1.3)
    plt.title("Demand break-even - first postponed orders")
    plt.xlabel("Demand factor")
    plt.ylabel("Mean postponed orders")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_axis_1_abc_stacked(df):
    required = ["postponed_A", "postponed_B", "postponed_C"]
    missing = [column for column in required if column not in df.columns]

    if missing:
        print(
            "Skipping Axis 1 ABC stacked chart because columns are missing: "
            f"{', '.join(missing)}. Rerun Axis 1 to generate them."
        )
        return None

    path = PLOTS_DIR / "axis_1_postponed_orders_by_abc.png"
    summary = df.groupby("demand_factor")[required].mean().reset_index()
    x_labels = [f"{value:.1f}" for value in summary["demand_factor"]]
    bottom = [0] * len(summary)
    colours = {
        "postponed_A": "#B00032",
        "postponed_B": "#123C7C",
        "postponed_C": "#8AA6C8",
    }
    labels = {
        "postponed_A": "A",
        "postponed_B": "B",
        "postponed_C": "C",
    }

    plt.figure(figsize=(9, 5))

    for column in required:
        values = summary[column].tolist()
        plt.bar(
            x_labels,
            values,
            bottom=bottom,
            label=labels[column],
            color=colours[column],
        )
        bottom = [base + value for base, value in zip(bottom, values)]

    plt.title("Postponed orders by ABC category")
    plt.xlabel("Demand factor")
    plt.ylabel("Mean postponed orders")
    plt.legend(title="ABC category")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_axis_1_euros_per_kg(df):
    required = ["euros_produced", "kilos"]
    missing = [column for column in required if column not in df.columns]

    if missing:
        print(
            "Skipping Axis 1 euros/kg chart because columns are missing: "
            f"{', '.join(missing)}."
        )
        return None

    path = PLOTS_DIR / "axis_1_euros_per_kg.png"
    temp = df.copy()
    temp["euros_per_kg"] = temp.apply(
        lambda row: (
            row["euros_produced"] / row["kilos"]
            if row["kilos"] and row["kilos"] > 0
            else 0
        ),
        axis=1,
    )
    summary = aggregate_by(temp, "demand_factor", ["euros_per_kg"])
    x = summary["demand_factor"]
    mean = summary["euros_per_kg_mean"]
    std = summary["euros_per_kg_std"].fillna(0)

    plt.figure(figsize=(9, 5))
    plt.plot(x, mean, marker="o", color="#123C7C", linewidth=2)
    plt.fill_between(
        x,
        mean - std,
        mean + std,
        color="#123C7C",
        alpha=0.18,
        linewidth=0,
    )
    plt.title("Demand scaling - Economic value per kg produced")
    plt.xlabel("Demand factor")
    plt.ylabel("Economic value per kg")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_axis_1_postponed_by_abc_lines(df):
    columns = ["postponed_A", "postponed_B", "postponed_C"]
    missing = [column for column in columns if column not in df.columns]

    if missing:
        print(
            "Skipping Axis 1 ABC line chart because columns are missing: "
            f"{', '.join(missing)}. Rerun Axis 1 to generate them."
        )
        return None

    path = PLOTS_DIR / "axis_1_postponed_by_abc_lines.png"
    colours = {
        "postponed_A": "#B00032",
        "postponed_B": "#123C7C",
        "postponed_C": "#8AA6C8",
    }
    labels = {
        "postponed_A": "A",
        "postponed_B": "B",
        "postponed_C": "C",
    }

    plt.figure(figsize=(9, 5))

    for column in columns:
        summary = aggregate_by(df, "demand_factor", [column])
        x = summary["demand_factor"]
        mean = summary[f"{column}_mean"]
        std = summary[f"{column}_std"].fillna(0)

        plt.plot(
            x,
            mean,
            marker="o",
            color=colours[column],
            linewidth=2,
            label=labels[column],
        )
        plt.fill_between(
            x,
            mean - std,
            mean + std,
            color=colours[column],
            alpha=0.14,
            linewidth=0,
        )

    plt.title("Demand scaling - Postponed orders by ABC category")
    plt.xlabel("Demand factor")
    plt.ylabel("Mean postponed orders")
    plt.legend(title="ABC category")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_axis_1_setup_time_paradox(df):
    if "total_setup_time" not in df.columns:
        print(
            "Skipping Axis 1 setup time paradox chart because "
            "total_setup_time is missing."
        )
        return None

    path = PLOTS_DIR / "axis_1_setup_time_paradox.png"
    summary = aggregate_by(df, "demand_factor", ["total_setup_time"])
    x = summary["demand_factor"]
    mean = summary["total_setup_time_mean"]
    std = summary["total_setup_time_std"].fillna(0)

    plt.figure(figsize=(9, 5))
    plt.plot(x, mean, marker="o", color="#725AC1", linewidth=2)
    plt.fill_between(
        x,
        mean - std,
        mean + std,
        color="#725AC1",
        alpha=0.18,
        linewidth=0,
    )

    if len(x) > 1:
        annotation_x = x.iloc[max(0, len(x) // 2)]
        annotation_y = mean.iloc[max(0, len(mean) // 2)]
        plt.annotate(
            "Setup time decreases as demand increases - "
            "denser schedules reduce product switching",
            xy=(annotation_x, annotation_y),
            xytext=(0.62, 0.82),
            textcoords="axes fraction",
            arrowprops={
                "arrowstyle": "->",
                "color": "#111111",
                "linewidth": 1.2,
            },
            fontsize=9,
            ha="left",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#C9D5E8",
                "alpha": 0.95,
            },
        )

    plt.title("Demand scaling - Setup time paradox")
    plt.xlabel("Demand factor")
    plt.ylabel("Setup time")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_axis_2_marginal_returns(df):
    path = PLOTS_DIR / "axis_2_marginal_returns.png"
    summary = (
        df.groupby("total_operators")[["euros_produced", "postponed_orders"]]
        .mean()
        .reset_index()
        .sort_values("total_operators")
    )
    summary["delta_euros"] = summary["euros_produced"].diff()
    summary["delta_postponed_orders"] = summary["postponed_orders"].diff()
    saturation_operator = None
    previous_gain = None

    for _, row in summary.dropna(subset=["delta_euros"]).iterrows():
        current_gain = abs(row["delta_euros"])

        if previous_gain is not None and previous_gain > 0:
            if current_gain < 0.05 * previous_gain:
                saturation_operator = row["total_operators"]
                break

        previous_gain = current_gain

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(
        summary["total_operators"],
        summary["delta_euros"],
        marker="o",
        color="#123C7C",
        linewidth=2,
    )
    axes[0].set_ylabel("Delta economic value")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(
        summary["total_operators"],
        summary["delta_postponed_orders"],
        marker="o",
        color="#B00032",
        linewidth=2,
    )
    axes[1].set_ylabel("Delta postponed orders")
    axes[1].set_xlabel("Total operators")
    axes[1].grid(True, alpha=0.25)

    if saturation_operator is not None:
        for ax in axes:
            ax.axvline(
                saturation_operator,
                color="#111111",
                linestyle="--",
                linewidth=1.2,
            )
        axes[0].text(
            saturation_operator,
            axes[0].get_ylim()[1],
            f"Saturation: {int(saturation_operator)}",
            ha="right",
            va="top",
            rotation=90,
            color="#111111",
        )

    fig.suptitle("Operator sweep - marginal returns")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_axis_2_boxplots(df):
    created = []

    for metric in ["euros_produced", "postponed_orders"]:
        path = PLOTS_DIR / f"axis_2_boxplot_{metric}.png"
        ordered_counts = sorted(df["total_operators"].unique())
        data = [
            df[df["total_operators"] == count][metric].tolist()
            for count in ordered_counts
        ]

        plt.figure(figsize=(11, 5))
        plt.boxplot(data, labels=[str(count) for count in ordered_counts])
        plt.title(f"Operator sweep distribution - {METRIC_LABELS[metric]}")
        plt.xlabel("Total operators")
        plt.ylabel(METRIC_LABELS[metric])
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()
        created.append(path)

    return created


def normalised_axis_3_summary(df, metrics):
    baseline = baseline_metric_means(df)
    scenario_df = df[df["weight_name"] != "baseline"].copy()
    grouped = (
        scenario_df
        .groupby(["weight_name", "weight_value"])[metrics]
        .mean()
        .reset_index()
    )

    for metric in metrics:
        grouped[f"{metric}_normalised"] = (
            grouped[metric] / baseline.get(metric, 1)
        )

    return grouped


def plot_axis_3_parallel_coordinates(df):
    metrics = [
        "postponed_orders",
        "postponed_boxes",
        "euros_produced",
        "total_setup_time",
        "operators_used",
    ]
    summary = normalised_axis_3_summary(df, metrics)
    path = PLOTS_DIR / "axis_3_parallel_coordinates.png"
    axis_columns = [f"{metric}_normalised" for metric in metrics]
    positions = list(range(len(axis_columns)))
    colour_map = plt.cm.get_cmap("tab10")
    weight_names = list(summary["weight_name"].drop_duplicates())
    colours = {
        weight_name: colour_map(index % 10)
        for index, weight_name in enumerate(weight_names)
    }

    plt.figure(figsize=(12, 6))

    for _, row in summary.iterrows():
        values = [row[column] for column in axis_columns]
        plt.plot(
            positions,
            values,
            color=colours[row["weight_name"]],
            alpha=0.35,
            linewidth=1.2,
        )

    legend_handles = [
        plt.Line2D([0], [0], color=colours[name], linewidth=2, label=WEIGHT_LABELS.get(name, name))
        for name in weight_names
    ]
    plt.xticks(
        positions,
        [METRIC_LABELS[metric] for metric in metrics],
        rotation=25,
        ha="right",
    )
    plt.ylabel("Metric value / baseline value")
    plt.title("FO weight sensitivity - parallel coordinates")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(handles=legend_handles, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_axis_3_euros_heatmap(df):
    path = PLOTS_DIR / "axis_3_euros_heatmap.png"
    baseline = baseline_metric_means(df).get("euros_produced", 1)
    scenario_df = df[df["weight_name"] != "baseline"].copy()
    summary = (
        scenario_df
        .groupby(["weight_name", "weight_value"])["euros_produced"]
        .mean()
        .reset_index()
    )
    summary["euros_normalised"] = summary["euros_produced"] / baseline
    weight_names = list(DEFAULT_NORMALISED_WEIGHTS.keys())
    weight_values = WEIGHT_VALUES
    matrix = []

    for weight_name in weight_names:
        row = []
        for weight_value in weight_values:
            match = summary[
                (summary["weight_name"] == weight_name)
                & (summary["weight_value"] == weight_value)
            ]
            row.append(match["euros_normalised"].iloc[0] if not match.empty else math.nan)
        matrix.append(row)

    plt.figure(figsize=(12, 5.5))
    image = plt.imshow(matrix, aspect="auto", cmap="Blues")
    plt.colorbar(image, label="Economic value / baseline")
    plt.xticks(
        range(len(weight_values)),
        [f"{value:.1f}" for value in weight_values],
    )
    plt.yticks(
        range(len(weight_names)),
        [WEIGHT_LABELS.get(name, name) for name in weight_names],
    )
    plt.xlabel("Weight value")
    plt.ylabel("Weight being varied")
    plt.title("FO weight sensitivity - economic value heatmap")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_combined_summary():
    axis_1_csv = RESULTS_DIR / "axis_1.csv"
    axis_2_csv = RESULTS_DIR / "axis_2.csv"
    axis_3_csv = RESULTS_DIR / "axis_3.csv"

    if not (axis_1_csv.exists() and axis_2_csv.exists() and axis_3_csv.exists()):
        print(
            "Skipping combined summary figure because one or more axis CSVs are missing."
        )
        return None

    axis_1 = pd.read_csv(axis_1_csv)
    axis_2 = pd.read_csv(axis_2_csv)
    axis_3 = pd.read_csv(axis_3_csv)
    path = PLOTS_DIR / "combined_scenario_summary.png"
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, df, x_col, title, xlabel in [
        (axes[0], axis_1, "demand_factor", "Axis 1 - Demand scaling", "Demand factor"),
        (axes[1], axis_2, "total_operators", "Axis 2 - Operator sweep", "Total operators"),
    ]:
        summary = aggregate_by(df, x_col, ["euros_produced"])
        ax.plot(
            summary[x_col],
            summary["euros_produced_mean"],
            marker="o",
            color="#123C7C",
            linewidth=2,
        )
        ax.fill_between(
            summary[x_col],
            summary["euros_produced_mean"] - summary["euros_produced_std"].fillna(0),
            summary["euros_produced_mean"] + summary["euros_produced_std"].fillna(0),
            color="#123C7C",
            alpha=0.18,
            linewidth=0,
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Economic value produced")
        ax.grid(True, alpha=0.25)

    scenario_axis_3 = axis_3[axis_3["weight_name"] != "baseline"].copy()
    for weight_name, weight_df in scenario_axis_3.groupby("weight_name"):
        summary = aggregate_by(weight_df, "weight_value", ["euros_produced"])
        axes[2].plot(
            summary["weight_value"],
            summary["euros_produced_mean"],
            marker="o",
            linewidth=1.5,
            label=WEIGHT_LABELS.get(weight_name, weight_name),
        )

    axes[2].set_title("Axis 3 - FO weight sensitivity")
    axes[2].set_xlabel("Weight value")
    axes[2].set_ylabel("Economic value produced")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(fontsize=8)
    fig.suptitle("Scenario analysis overview - economic value produced")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_axis_1(csv_path):
    df = pd.read_csv(csv_path)
    created = []
    metrics = [
        "postponed_orders",
        "postponed_boxes",
        "euros_produced",
        "fulfillment_rate",
        "fitness",
        "kilos",
    ]

    for metric in metrics:
        path = PLOTS_DIR / f"axis_1_demand_scaling_{metric}.png"
        created.append(
            save_line_plot_with_std(
                df,
                "demand_factor",
                metric,
                f"Demand scaling - {METRIC_LABELS[metric]}",
                "Demand factor",
                METRIC_LABELS[metric],
                path,
                vertical_line=1.0,
                vertical_label="Baseline",
            )
        )

    created.append(plot_axis_1_break_even(df))
    abc_plot = plot_axis_1_abc_stacked(df)

    if abc_plot is not None:
        created.append(abc_plot)

    for extra_plot in [
        plot_axis_1_euros_per_kg(df),
        plot_axis_1_postponed_by_abc_lines(df),
        plot_axis_1_setup_time_paradox(df),
    ]:
        if extra_plot is not None:
            created.append(extra_plot)

    return created


def plot_axis_2(csv_path, baseline_operators):
    df = pd.read_csv(csv_path)
    created = []
    metrics = [
        "postponed_orders",
        "postponed_boxes",
        "euros_produced",
        "fitness",
        "kilos",
    ]

    for metric in metrics:
        path = PLOTS_DIR / f"axis_2_operator_sweep_{metric}.png"
        created.append(
            save_line_plot_with_std(
                df,
                "total_operators",
                metric,
                f"Operator sweep - {METRIC_LABELS[metric]}",
                "Total operators",
                METRIC_LABELS[metric],
                path,
                vertical_line=baseline_operators,
                vertical_label="Real June baseline",
            )
        )

    created.append(plot_axis_2_marginal_returns(df))
    created.extend(plot_axis_2_boxplots(df))

    return created


def baseline_metric_means(df):
    baseline = df[df["weight_name"] == "baseline"]

    if baseline.empty:
        return {}

    metrics = [
        "postponed_orders",
        "postponed_boxes",
        "euros_produced",
        "total_setup_time",
        "operators_used",
        "kilos",
    ]
    return {
        metric: max(1e-9, baseline[metric].mean())
        for metric in metrics
    }


def plot_axis_3_metric_panels(weight_name, weight_df):
    panel_metrics = [
        ("fulfillment_rate", "Fulfilment Rate (%)", 100),
        ("postponed_orders", "Postponed Orders", 1),
        ("total_setup_time", "Total Setup Time (min)", 1),
        ("euros_produced", "Revenue Produced (€)", 1),
    ]
    optional_metrics = [
        ("delay_days_total", "Total Delay (days)", 1),
        ("capacity_utilisation", "Capacity Utilisation (%)", 100),
    ]
    panel_metrics.extend(
        metric
        for metric in optional_metrics
        if metric[0] in weight_df.columns
    )
    label = WEIGHT_LABELS.get(weight_name, weight_name)
    n_columns = 3 if len(panel_metrics) > 4 else 4
    n_rows = math.ceil(len(panel_metrics) / n_columns)
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(5.2 * n_columns, 4.5 * n_rows),
        squeeze=False,
    )

    for ax, (metric, y_label, scale) in zip(axes.flat, panel_metrics):
        summary = aggregate_by(weight_df, "weight_value", [metric])
        x = summary["weight_value"]
        mean = summary[f"{metric}_mean"] * scale
        std = summary[f"{metric}_std"].fillna(0) * scale

        ax.plot(
            x,
            mean,
            marker="o",
            linewidth=1.7,
            color="#0879bd",
        )
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color="#0879bd",
            alpha=0.18,
            linewidth=0,
        )
        ax.set_xlabel("Weight value")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)

    for ax in axes.flat[len(panel_metrics):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Axis 3 — Sensitivity to weight — {label}",
        fontweight="bold",
    )
    fig.tight_layout()
    path = PLOTS_DIR / f"axis_3_weight_panels_{weight_name}.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_axis_3(csv_path):
    df = pd.read_csv(csv_path)
    created = []
    baseline_means = baseline_metric_means(df)
    metrics = [
        "postponed_orders",
        "postponed_boxes",
        "euros_produced",
        "total_setup_time",
        "operators_used",
        "kilos",
    ]
    colours = {
        "postponed_orders": "#B00032",
        "postponed_boxes": "#D64C2A",
        "euros_produced": "#123C7C",
        "total_setup_time": "#725AC1",
        "operators_used": "#2F7D32",
        "kilos": "#0088A8",
    }

    scenario_df = df[df["weight_name"] != "baseline"].copy()

    for weight_name, weight_df in scenario_df.groupby("weight_name"):
        plt.figure(figsize=(11, 6))

        for metric in metrics:
            baseline = baseline_means.get(metric, 1)
            temp = weight_df.copy()
            temp[f"{metric}_normalised"] = temp[metric] / baseline
            summary = aggregate_by(temp, "weight_value", [f"{metric}_normalised"])
            x = summary["weight_value"]
            mean = summary[f"{metric}_normalised_mean"]
            std = summary[f"{metric}_normalised_std"].fillna(0)

            plt.plot(
                x,
                mean,
                marker="o",
                linewidth=1.7,
                label=METRIC_LABELS[metric],
                color=colours[metric],
            )
            plt.fill_between(
                x,
                mean - std,
                mean + std,
                color=colours[metric],
                alpha=0.10,
                linewidth=0,
            )

        default_value = DEFAULT_NORMALISED_WEIGHTS.get(weight_name)

        if default_value is not None:
            plt.axvline(default_value, color="#111111", linestyle="--", linewidth=1.2)
            plt.text(
                default_value,
                plt.ylim()[1],
                "Default",
                color="#111111",
                ha="right",
                va="top",
                rotation=90,
            )

        label = WEIGHT_LABELS.get(weight_name, weight_name)
        plt.title(f"FO weight sensitivity - {label}")
        plt.xlabel(f"{label} weight")
        plt.ylabel("Metric value / baseline value")
        plt.grid(True, alpha=0.25)
        plt.legend(ncol=2)
        plt.tight_layout()
        path = PLOTS_DIR / f"axis_3_weight_sensitivity_{weight_name}.png"
        plt.savefig(path, dpi=300)
        plt.close()
        created.append(path)
        created.append(
            plot_axis_3_metric_panels(weight_name, weight_df)
        )

    return created


def run_axis_1(excel_path, max_workers):
    csv_path = RESULTS_DIR / "axis_1.csv"

    if csv_path.exists():
        print(f"Axis 1 GA skipped because {csv_path} already exists.")
        return [], plot_axis_1(csv_path)

    jobs = [
        (excel_path, demand_factor, seed)
        for demand_factor in DEMAND_FACTORS
        for seed in SEEDS
    ]
    rows = execute_parallel(jobs, run_axis_1_case, max_workers=max_workers)
    raw_csv = save_axis_csv(1, rows)
    df = pd.DataFrame(rows)
    summary_csv = save_summary_csv(
        1,
        df,
        ["demand_factor"],
        [
            "postponed_orders",
            "postponed_boxes",
            "euros_produced",
            "fulfillment_rate",
            "fitness",
            "kilos",
            "postponed_A",
            "postponed_B",
            "postponed_C",
        ],
    )
    csv_created = [raw_csv, summary_csv]
    plots_created = plot_axis_1(raw_csv)
    return csv_created, plots_created


def run_axis_2(excel_path, max_workers):
    csv_path = RESULTS_DIR / "axis_2.csv"

    if csv_path.exists():
        print(f"Axis 2 GA skipped because {csv_path} already exists.")
        base_instance = load_base_instance(excel_path)
        baseline_operators = base_instance.get("standard_operators", 20)
        return [], plot_axis_2(csv_path, baseline_operators)

    jobs = [
        (excel_path, total_operators, seed)
        for total_operators in OPERATOR_COUNTS
        for seed in SEEDS
    ]
    rows = execute_parallel(jobs, run_axis_2_case, max_workers=max_workers)
    raw_csv = save_axis_csv(2, rows)
    df = pd.DataFrame(rows)
    summary_csv = save_summary_csv(
        2,
        df,
        ["total_operators"],
        [
            "postponed_orders",
            "postponed_boxes",
            "euros_produced",
            "fitness",
            "kilos",
        ],
    )
    csv_created = [raw_csv, summary_csv]

    base_instance = load_base_instance(excel_path)
    baseline_operators = base_instance.get("standard_operators", 20)
    plots_created = plot_axis_2(raw_csv, baseline_operators)
    return csv_created, plots_created


def run_axis_3(excel_path, max_workers, weight_index=None):
    if weight_index is None:
        csv_path = RESULTS_DIR / "axis_3.csv"
    else:
        csv_path = RESULTS_DIR / f"axis_3_weight_{weight_index}.csv"

    if csv_path.exists():
        print(f"Axis 3 GA skipped because {csv_path} already exists.")
        return [], plot_axis_3(csv_path)

    jobs = []

    for seed in SEEDS:
        jobs.append((excel_path, "baseline", 0.0, seed))

    weight_names = list(DEFAULT_NORMALISED_WEIGHTS)

    if weight_index is not None:
        weight_names = [weight_names[weight_index]]

    for weight_name in weight_names:
        for weight_value in WEIGHT_VALUES:
            for seed in SEEDS:
                jobs.append((excel_path, weight_name, weight_value, seed))

    rows = execute_parallel(jobs, run_axis_3_case, max_workers=max_workers)
    df = pd.DataFrame(rows)

    if weight_index is None:
        raw_csv = save_axis_csv(3, rows)
    else:
        raw_csv = csv_path
        df.to_csv(raw_csv, index=False)

    csv_created = [raw_csv]

    if weight_index is None:
        summary_csv = save_summary_csv(
            3,
            df,
            ["weight_name", "weight_value"],
            [
                "postponed_orders",
                "postponed_boxes",
                "euros_produced",
                "total_setup_time",
                "operators_used",
                "delay_days_total",
                "capacity_utilisation",
                "kilos",
            ],
        )
        csv_created.append(summary_csv)

    plots_created = plot_axis_3(raw_csv)
    return csv_created, plots_created


def merge_axis_3_weight_csvs():
    frames = []

    for weight_index in range(6):
        path = RESULTS_DIR / f"axis_3_weight_{weight_index}.csv"

        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

        frames.append(pd.read_csv(path))

    merged = pd.concat(frames, ignore_index=True)
    output_path = RESULTS_DIR / "axis_3.csv"
    merged.to_csv(output_path, index=False)
    return output_path


def regenerate_plots(axis, excel_path):
    created = []

    if axis in [None, "1"]:
        csv_path = RESULTS_DIR / "axis_1.csv"
        if csv_path.exists():
            created.extend(plot_axis_1(csv_path))
        else:
            print(f"Cannot regenerate Axis 1 plots: missing {csv_path}")

    if axis in [None, "2"]:
        csv_path = RESULTS_DIR / "axis_2.csv"
        if csv_path.exists():
            base_instance = load_base_instance(excel_path)
            baseline_operators = base_instance.get("standard_operators", 20)
            created.extend(plot_axis_2(csv_path, baseline_operators))
        else:
            print(f"Cannot regenerate Axis 2 plots: missing {csv_path}")

    if axis in [None, "3"]:
        csv_path = RESULTS_DIR / "axis_3.csv"
        if csv_path.exists():
            created.extend(plot_axis_3(csv_path))
        else:
            print(f"Cannot regenerate Axis 3 plots: missing {csv_path}")

    combined = plot_combined_summary()

    if combined is not None:
        created.append(combined)

    return created


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run unattended scenario analysis for the production GA."
    )
    parser.add_argument(
        "--axis",
        choices=["1", "2", "3"],
        default=None,
        help="Scenario axis to run. If omitted, all axes are run.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate plots from existing CSVs without rerunning the GA.",
    )
    parser.add_argument(
        "--weight-index",
        type=int,
        default=None,
        choices=[0, 1, 2, 3, 4, 5],
        help="Run only this weight index for axis 3. Saves to axis_3_weight_N.csv.",
    )
    parser.add_argument(
        "--merge-weights",
        action="store_true",
        help="Merge axis_3_weight_0..4.csv into axis_3.csv before plotting.",
    )
    parser.add_argument(
        "--excel-path",
        default=str(DEFAULT_EXCEL_PATH),
        help="Path to the Excel input file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory where scenario analysis CSVs and plots are saved.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of parallel processes. Defaults to available CPU count.",
    )
    parser.add_argument(
        "--planning-start",
        default=None,
        help="Planning horizon start date (YYYY-MM-DD). Defaults to the first day of the demand month.",
    )
    parser.add_argument(
        "--planning-end",
        default=None,
        help="Planning horizon end date (YYYY-MM-DD). Defaults to the last day of the demand month.",
    )
    parser.add_argument("--operators", type=int, default=18)
    parser.add_argument("--shifts", type=int, choices=[1, 2], default=1)
    parser.add_argument("--shift-start", default="08:00")
    parser.add_argument("--shift-end", default="16:30")
    parser.add_argument("--lunch-minutes", type=int, default=30)
    parser.add_argument("--cleaning-minutes", type=int, default=30)
    parser.add_argument(
        "--non-working-dates",
        default="",
        help="Comma-separated non-working dates in YYYY-MM-DD format.",
    )
    return parser.parse_args()


def main():
    start_time = time.perf_counter()
    args = parse_args()
    configure_output_dir(args.output_dir)
    configure_operational_inputs(args)
    ensure_output_dirs()

    excel_path = Path(args.excel_path).resolve()
    created_csvs = []
    created_plots = []

    if args.plots_only:
        if args.axis == "3" and args.merge_weights:
            merge_axis_3_weight_csvs()

        created_plots = regenerate_plots(args.axis, excel_path)
    else:
        axes_to_run = [args.axis] if args.axis else ["1", "2", "3"]

        for axis in axes_to_run:
            if axis == "1":
                csvs, plots = run_axis_1(excel_path, args.max_workers)
            elif axis == "2":
                csvs, plots = run_axis_2(excel_path, args.max_workers)
            elif axis == "3":
                csvs, plots = run_axis_3(
                    excel_path,
                    args.max_workers,
                    weight_index=args.weight_index,
                )
            else:
                csvs, plots = [], []

            created_csvs.extend(csvs)
            created_plots.extend(plots)

        combined = plot_combined_summary()

        if combined is not None:
            created_plots.append(combined)

    elapsed = time.perf_counter() - start_time
    print("\n=== SCENARIO ANALYSIS COMPLETE ===")
    print(f"Total time elapsed: {elapsed / 60:.2f} minutes")

    if created_csvs:
        print("\nCSV files created:")
        for path in created_csvs:
            print(f"  {path}")
    else:
        print("\nCSV files created: none")

    if created_plots:
        print("\nPNG files created:")
        for path in created_plots:
            print(f"  {path}")
    else:
        print("\nPNG files created: none")


if __name__ == "__main__":
    main()
