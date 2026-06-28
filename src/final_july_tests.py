import argparse
import copy
import csv
import pathlib
import statistics
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from evaluator import (
    DEFAULT_NORMALISED_WEIGHTS,
    evaluate_solution,
    get_available_line_time_for_day,
)
from generate_instance import load_real_instance
from geneticalgorithm import run_genetic_algorithm


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_INSTANCE_FILE = PROJECT_DIR / "Inputs_July.xlsx"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "final_july_tests"

SEEDS = [42, 43, 44]
DEMAND_FACTORS = [0.6, 0.7, 0.8, 0.9, 1.0]
OPERATOR_COUNTS = list(range(18, 31))
WEIGHT_VALUES = [round(index / 10, 1) for index in range(11)]
WEIGHT_NAMES = [
    "postponement",
    "delay",
    "setup",
    "economic_value",
    "capacity_utilisation",
    "operator_utilisation",
]
WEIGHT_LABELS = {
    "postponement": "Postponement",
    "delay": "Delay",
    "setup": "Setup",
    "economic_value": "Economic value",
    "capacity_utilisation": "Capacity utilisation",
    "operator_utilisation": "Operator utilisation",
}

GA_PARAMETERS = {
    "population_size": 108,
    "generations": 200,
    "mutation_rate": 0.057,
    "elite_size": 5,
    "tournament_size": 3,
    "stagnation_k": 26,
}

BASELINE_OPERATIONAL_CONFIG = {
    "shift_start_min": 8 * 60,
    "shift_end_min": 16 * 60 + 30,
    "lunch_break_min": 30,
    "cleaning_time_min": 30,
    "standard_operators": 20,
    "operators": 20,
    "shifts": 1,
}


def load_july_instance(instance_file, operators=20):
    config = dict(BASELINE_OPERATIONAL_CONFIG)
    config["standard_operators"] = operators
    config["operators"] = operators
    return load_real_instance(
        str(instance_file),
        operational_config=config,
    )


def scale_demand(instance, factor):
    scenario = copy.deepcopy(instance)
    for order in scenario.get("demand", []):
        original_quantity = order.get("master_boxes", 0) or 0
        order["master_boxes"] = (
            0
            if original_quantity <= 0
            else max(1, int(round(original_quantity * factor)))
        )
    scenario.setdefault("_meta", {})["demand_factor"] = factor
    return scenario


def set_uniform_operators(instance, operators):
    scenario = copy.deepcopy(instance)
    scenario["standard_operators"] = int(operators)
    scenario["standard_operators_by_day"] = {
        day: int(operators)
        for day in range(1, scenario.get("n_days", 0) + 1)
    }
    scenario.setdefault("_meta", {})["operators"] = int(operators)
    return scenario


def normalise_weights(weights):
    total = sum(max(0, value) for value in weights.values())
    if total <= 0:
        return dict(DEFAULT_NORMALISED_WEIGHTS)
    return {key: max(0, value) / total for key, value in weights.items()}


def make_weight_sensitivity_weights(weight_name, tested_value):
    base_weights = dict(DEFAULT_NORMALISED_WEIGHTS)
    other_names = [name for name in WEIGHT_NAMES if name != weight_name]
    remaining_weight = max(0.0, 1.0 - tested_value)
    other_total = sum(base_weights[name] for name in other_names)

    weights = {weight_name: tested_value}
    for name in other_names:
        weights[name] = remaining_weight * base_weights[name] / other_total

    return normalise_weights(weights)


def make_two_shift_slot_instance(instance):
    """Represent each physical shift as a separate schedulable slot.

    Slot 1 = day 1 shift 1, slot 2 = day 1 shift 2, slot 3 = day 2 shift 1, etc.
    This lets the GA assign an order to either shift while keeping each shift's
    capacity and 20-operator pool independent.
    """
    shifts_per_day = 2
    scenario = copy.deepcopy(instance)
    original_n_days = instance["n_days"]
    scenario["n_days"] = original_n_days * shifts_per_day
    scenario["days"] = [
        f"day_{day}_shift_{shift}"
        for day in range(1, original_n_days + 1)
        for shift in range(1, shifts_per_day + 1)
    ]
    scenario["working_days"] = [
        working_day
        for working_day in instance.get("working_days", [])
        for _ in range(shifts_per_day)
    ]
    scenario["monday_days"] = [
        (day - 1) * shifts_per_day + shift
        for day in instance.get("monday_days", [])
        for shift in range(1, shifts_per_day + 1)
    ]

    scenario["standard_operators_by_day"] = {}
    scenario["daily_capacity_min"] = {}
    scenario["daily_shift_start_min"] = {}
    scenario["daily_shift_end_min"] = {}
    scenario["daily_shifts"] = {}

    for day in range(1, original_n_days + 1):
        first_start = instance.get("daily_shift_start_min", {}).get(day, 8 * 60)
        first_end = instance.get("daily_shift_end_min", {}).get(day, 16 * 60 + 30)
        gross_shift_duration = first_end - first_start
        available_capacity = instance.get("daily_capacity_min", {}).get(
            day,
            instance.get("available_line_time_min", 450),
        )

        for shift in range(1, shifts_per_day + 1):
            slot = (day - 1) * shifts_per_day + shift
            start = first_start + (shift - 1) * gross_shift_duration
            scenario["standard_operators_by_day"][slot] = instance.get(
                "standard_operators",
                20,
            )
            scenario["daily_capacity_min"][slot] = available_capacity
            scenario["daily_shift_start_min"][slot] = start
            scenario["daily_shift_end_min"][slot] = start + gross_shift_duration
            scenario["daily_shifts"][slot] = 1

    for order in scenario.get("demand", []):
        if order.get("delivery_date") is not None:
            order["delivery_date"] = int(order["delivery_date"]) * shifts_per_day

    scenario.setdefault("_meta", {})["shift_slots_per_day"] = shifts_per_day
    return scenario


def run_ga_case(instance, seed, objective_weights=None):
    solution, metrics, actual_generations = run_genetic_algorithm(
        instance,
        seed=seed,
        verbose=False,
        objective_weights=objective_weights,
        **GA_PARAMETERS,
    )
    metrics["actual_generations"] = actual_generations
    return solution, metrics


def total_demand_boxes(instance):
    return sum(order.get("master_boxes", 0) or 0 for order in instance.get("demand", []))


def extract_kpis(metrics, instance):
    total_orders = len(instance.get("demand", []))
    postponed_orders = metrics.get("postponed_orders", 0)
    demand_boxes = max(1, total_demand_boxes(instance))
    postponed_boxes = metrics.get("postponed_boxes", 0) or 0

    return {
        "fitness": metrics.get("normalised_fitness"),
        "postponed_orders": postponed_orders,
        "postponed_boxes": postponed_boxes,
        "fulfilment_rate_pct": (
            (total_orders - postponed_orders) / total_orders * 100
            if total_orders
            else 0
        ),
        "box_fulfilment_rate_pct": (1 - postponed_boxes / demand_boxes) * 100,
        "scheduled_kg": metrics.get("scheduled_kg", 0),
        "scheduled_economic_value_eur": metrics.get("scheduled_economic_value", 0),
        "setup_time_min": metrics.get("setup_total_min", 0),
        "operator_utilisation_min": metrics.get("operator_usage_minutes", 0),
        "capacity_violations": metrics.get("capacity_violations", 0),
        "operator_violations": metrics.get("operator_violations", 0),
        "peak_operators": metrics.get("peak_operators", 0),
        "actual_generations": metrics.get("actual_generations", 0),
    }


def slot_to_day(slot, shifts_per_day):
    return ((slot - 1) // shifts_per_day) + 1


def slot_to_shift(slot, shifts_per_day):
    return ((slot - 1) % shifts_per_day) + 1


def shift_utilisation_rows(instance, solution, scenario, seed, shifts_per_day):
    metrics = evaluate_solution(solution, instance)
    production_by_slot_line = metrics.get("production_time_by_day_line", {})
    operator_intervals = metrics.get("operator_usage_intervals", [])
    rows = []

    for slot in range(1, instance["n_days"] + 1):
        calendar_day = slot_to_day(slot, shifts_per_day)
        shift = slot_to_shift(slot, shifts_per_day)
        available = get_available_line_time_for_day(instance, slot)
        line_count = len(instance.get("final_lines", []))
        total_available = available * line_count
        total_production = sum(
            production_by_slot_line.get((slot, line), 0)
            for line in instance.get("final_lines", [])
        )
        slot_intervals = [
            interval
            for interval in operator_intervals
            if interval.get("day") == slot
        ]
        operator_minutes = sum(
            max(0, interval.get("end", 0) - interval.get("start", 0))
            * interval.get("operators", 0)
            for interval in slot_intervals
        )
        peak_operators = max(
            [interval.get("operators", 0) for interval in slot_intervals],
            default=0,
        )
        avg_operators = operator_minutes / available if available else 0

        rows.append({
            "scenario": scenario,
            "seed": seed,
            "calendar_day": calendar_day,
            "shift": shift,
            "slot": slot,
            "line_count": line_count,
            "production_time_min": total_production,
            "available_time_min": total_available,
            "capacity_utilisation_pct": (
                total_production / total_available * 100
                if total_available
                else 0
            ),
            "operator_minutes": operator_minutes,
            "avg_operators_used": avg_operators,
            "peak_operators_used": peak_operators,
            "active_shift": total_production > 0,
        })

    return rows


def mean(values):
    clean_values = [
        value
        for value in values
        if value is not None and value != ""
    ]
    return statistics.mean(clean_values) if clean_values else ""


def summarise(rows, group_columns):
    metric_columns = [
        "fitness",
        "postponed_orders",
        "postponed_boxes",
        "fulfilment_rate_pct",
        "box_fulfilment_rate_pct",
        "scheduled_kg",
        "scheduled_economic_value_eur",
        "setup_time_min",
        "operator_utilisation_min",
        "capacity_violations",
        "operator_violations",
        "peak_operators",
        "actual_generations",
        "avg_capacity_utilisation_shift_1_pct",
        "avg_capacity_utilisation_shift_2_pct",
        "avg_operators_used_shift_1",
        "avg_operators_used_shift_2",
        "peak_operators_used_shift_1",
        "peak_operators_used_shift_2",
        "active_days_shift_1",
        "active_days_shift_2",
        "active_day_pct_shift_1",
        "active_day_pct_shift_2",
    ]
    grouped = {}
    for row in rows:
        key = tuple(row[column] for column in group_columns)
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for key, group_rows in sorted(grouped.items()):
        summary = {
            column: value
            for column, value in zip(group_columns, key)
        }
        summary["n_seeds"] = len(group_rows)
        for column in metric_columns:
            values = [row.get(column) for row in group_rows if column in row]
            summary[f"mean_{column}"] = mean(values)
        summary_rows.append(summary)
    return summary_rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def run_test_1(base_instance):
    rows = []
    for demand_factor in DEMAND_FACTORS:
        for seed in SEEDS:
            instance = scale_demand(base_instance, demand_factor)
            _solution, metrics = run_ga_case(instance, seed)
            row = {
                "test": "test_1_demand_scaling",
                "demand_factor": demand_factor,
                "seed": seed,
            }
            row.update(extract_kpis(metrics, instance))
            rows.append(row)
    return rows, summarise(rows, ["demand_factor"])


def run_test_2(base_instance):
    rows = []
    for operators in OPERATOR_COUNTS:
        for seed in SEEDS:
            instance = set_uniform_operators(base_instance, operators)
            _solution, metrics = run_ga_case(instance, seed)
            row = {
                "test": "test_2_operator_availability",
                "operators": operators,
                "seed": seed,
            }
            row.update(extract_kpis(metrics, instance))
            rows.append(row)
    return rows, summarise(rows, ["operators"])


def run_test_3(base_instance, weight_index=None):
    weight_names = WEIGHT_NAMES
    if weight_index is not None:
        weight_names = [WEIGHT_NAMES[weight_index]]

    rows = []
    for weight_name in weight_names:
        for tested_value in WEIGHT_VALUES:
            objective_weights = make_weight_sensitivity_weights(
                weight_name,
                tested_value,
            )
            for seed in SEEDS:
                solution, metrics = run_ga_case(
                    base_instance,
                    seed,
                    objective_weights=objective_weights,
                )
                row = {
                    "test": "test_3_weight_sensitivity",
                    "weight_name": weight_name,
                    "tested_weight_value": tested_value,
                    "seed": seed,
                }
                row.update(extract_kpis(metrics, base_instance))
                for used_name, used_value in objective_weights.items():
                    row[f"used_weight_{used_name}"] = used_value
                rows.append(row)
    return rows, summarise(rows, ["weight_name", "tested_weight_value"])


def run_test_4(base_instance):
    rows = []
    per_shift_rows = []
    scenarios = [
        ("one_shift_baseline", 1, base_instance),
        ("two_shift_slots", 2, make_two_shift_slot_instance(base_instance)),
    ]

    for scenario_name, shifts_per_day, instance in scenarios:
        for seed in SEEDS:
            solution, metrics = run_ga_case(instance, seed)
            row = {
                "test": "test_4_shift_structure",
                "scenario": scenario_name,
                "shifts_per_day": shifts_per_day,
                "seed": seed,
            }
            row.update(extract_kpis(metrics, instance))

            shift_rows = shift_utilisation_rows(
                instance,
                solution,
                scenario_name,
                seed,
                shifts_per_day,
            )
            per_shift_rows.extend(shift_rows)

            for shift in range(1, shifts_per_day + 1):
                shift_items = [
                    item
                    for item in shift_rows
                    if item["shift"] == shift
                ]
                row[f"avg_capacity_utilisation_shift_{shift}_pct"] = mean(
                    [item["capacity_utilisation_pct"] for item in shift_items]
                )
                row[f"avg_operators_used_shift_{shift}"] = mean(
                    [item["avg_operators_used"] for item in shift_items]
                )
                row[f"peak_operators_used_shift_{shift}"] = max(
                    [item["peak_operators_used"] for item in shift_items],
                    default=0,
                )
                active_days = len({
                    item["calendar_day"]
                    for item in shift_items
                    if item["active_shift"]
                })
                total_days = len({item["calendar_day"] for item in shift_items})
                row[f"active_days_shift_{shift}"] = active_days
                row[f"active_day_pct_shift_{shift}"] = (
                    active_days / total_days * 100
                    if total_days
                    else 0
                )
            rows.append(row)

    return rows, summarise(rows, ["scenario"]), per_shift_rows


def plot_line(df, x_col, y_col, path, title, x_label, y_label, group_col=None):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if group_col:
        for group, group_df in df.groupby(group_col):
            group_df = group_df.sort_values(x_col)
            ax.plot(group_df[x_col], group_df[y_col], marker="o", linewidth=2, label=str(group))
        ax.legend(title=group_col.replace("_", " ").title(), fontsize=8)
    else:
        df = df.sort_values(x_col)
        ax.plot(df[x_col], df[y_col], marker="o", linewidth=2, color="#153e7e")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_dual_axis(df, x_col, left_col, right_col, path, title, x_label, left_label, right_label):
    df = df.sort_values(x_col)
    fig, ax_left = plt.subplots(figsize=(8, 4.8))
    ax_right = ax_left.twinx()
    left_line = ax_left.plot(df[x_col], df[left_col], marker="o", color="#153e7e", label=left_label)
    right_line = ax_right.plot(df[x_col], df[right_col], marker="s", color="#b6003b", label=right_label)
    ax_left.set_title(title, fontweight="bold")
    ax_left.set_xlabel(x_label)
    ax_left.set_ylabel(left_label)
    ax_right.set_ylabel(right_label)
    ax_left.grid(True, alpha=0.25)
    lines = left_line + right_line
    labels = [line.get_label() for line in lines]
    ax_left.legend(lines, labels, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_grouped_bars(df, category_col, value_cols, path, title, y_label):
    labels = df[category_col].astype(str).tolist()
    x_positions = range(len(labels))
    width = 0.8 / max(1, len(value_cols))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#153e7e", "#b6003b", "#2f7d32", "#f59e0b"]
    for index, value_col in enumerate(value_cols):
        offset = (index - (len(value_cols) - 1) / 2) * width
        values = df[value_col].fillna(0).tolist()
        ax.bar(
            [position + offset for position in x_positions],
            values,
            width=width,
            label=value_col.replace("mean_", "").replace("_", " "),
            color=colors[index % len(colors)],
        )
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(y_label)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_heatmap(df, index_col, column_col, value_col, path, title, cbar_label):
    pivot = df.pivot_table(
        index=index_col,
        columns=column_col,
        values=value_col,
        aggfunc="mean",
    ).sort_index()
    fig, ax = plt.subplots(figsize=(8, max(4.8, len(pivot) * 0.28)))
    image = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(column_col.replace("_", " ").title())
    ax.set_ylabel(index_col.replace("_", " ").title())
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(col) for col in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(idx) for idx in pivot.index])
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_test3_weight_profiles(test_3, plots_dir, raw_test_3=None):
    metrics = [
        ("fitness", "mean_fitness", "Fitness", "Fitness"),
        ("box_fulfilment_rate_pct", "mean_box_fulfilment_rate_pct", "Box fulfilment rate", "Box Fulfilment Rate (%)"),
        ("postponed_boxes", "mean_postponed_boxes", "Postponed boxes", "Postponed Boxes"),
        ("scheduled_economic_value_eur", "mean_scheduled_economic_value_eur", "Economic value", "Revenue Produced (EUR)"),
        ("setup_time_min", "mean_setup_time_min", "Setup time", "Total Setup Time (min)"),
        ("operator_utilisation_min", "mean_operator_utilisation_min", "Operator use", "Operator-minutes"),
    ]

    for weight_name in WEIGHT_NAMES:
        weight_df = (
            test_3[test_3["weight_name"] == weight_name]
            .sort_values("tested_weight_value")
            .copy()
        )
        if weight_df.empty:
            continue

        raw_weight = None
        if raw_test_3 is not None:
            raw_weight = raw_test_3[raw_test_3["weight_name"] == weight_name].copy()

        fig, axes = plt.subplots(1, 6, figsize=(24, 4.8), sharex=True)
        default_value = DEFAULT_NORMALISED_WEIGHTS.get(weight_name, 0)

        for ax, (raw_col, mean_col, title, y_label) in zip(axes.flat, metrics):
            x_values = weight_df["tested_weight_value"]
            y_values = weight_df[mean_col]
            ax.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=2.4,
                color="#1f77b4",
            )
            if raw_weight is not None and raw_col in raw_weight.columns:
                spread = (
                    raw_weight
                    .groupby("tested_weight_value")[raw_col]
                    .agg(["min", "max"])
                    .reindex(x_values)
                    .reset_index()
                )
                ax.fill_between(
                    spread["tested_weight_value"].astype(float).to_numpy(),
                    spread["min"].astype(float).to_numpy(),
                    spread["max"].astype(float).to_numpy(),
                    color="#1f77b4",
                    alpha=0.18,
                    linewidth=0,
                )
            ax.axvline(
                default_value,
                color="#b6003b",
                linestyle="--",
                linewidth=1.4,
            )
            ax.set_title(title, fontweight="bold", fontsize=11)
            ax.set_xlabel("Weight value")
            ax.set_ylabel(y_label)
            ax.set_xlim(0, 1)
            ax.grid(True, linestyle="--", alpha=0.25)

        fig.suptitle(
            f"Test 3 - Sensitivity to {WEIGHT_LABELS.get(weight_name, weight_name)} weight",
            fontweight="bold",
            fontsize=15,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        fig.savefig(plots_dir / f"test3_weight_profile_{weight_name}.png", dpi=180)
        plt.close(fig)


def plot_test3_fulfilment_heatmap(test_3, plots_dir):
    pivot = (
        test_3.pivot_table(
            index="weight_name",
            columns="tested_weight_value",
            values="mean_box_fulfilment_rate_pct",
            aggfunc="mean",
        )
        .reindex(WEIGHT_NAMES)
        .sort_index(axis=1)
    )
    fig, ax = plt.subplots(figsize=(10, 4.8))
    image = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu")
    ax.set_title("Test 3 - Box fulfilment rate heatmap", fontweight="bold")
    ax.set_xlabel("Tested weight value")
    ax.set_ylabel("Weight")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{value:.1f}" for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([WEIGHT_LABELS.get(value, value) for value in pivot.index])
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Box fulfilment rate (%)")
    fig.tight_layout()
    fig.savefig(plots_dir / "test3_fulfilment_heatmap.png", dpi=180)
    plt.close(fig)


def plot_test3_sensitivity_range(test_3, plots_dir):
    ranges = (
        test_3.groupby("weight_name")["mean_box_fulfilment_rate_pct"]
        .agg(lambda values: values.max() - values.min())
        .reindex(WEIGHT_NAMES)
        .dropna()
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = [WEIGHT_LABELS.get(weight, weight) for weight in ranges.index]
    bars = ax.barh(labels, ranges.values, color="#153e7e")
    ax.set_title("Box Fulfilment Rate Sensitivity by Weight", fontweight="bold")
    ax.set_xlabel("Box fulfilment rate range (percentage points)")
    ax.grid(axis="x", alpha=0.25)

    for bar, value in zip(bars, ranges.values):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {value:.2f}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(plots_dir / "test3_sensitivity_range.png", dpi=180)
    plt.close(fig)


def plot_test4_normalised_comparison(test_4, plots_dir):
    metrics = [
        ("mean_box_fulfilment_rate_pct", "Box fulfilment rate"),
        ("mean_postponed_boxes", "Postponed boxes"),
        ("mean_scheduled_economic_value_eur", "Economic value"),
        ("mean_setup_time_min", "Setup time"),
        ("mean_operator_utilisation_min", "Operator utilisation"),
    ]
    baseline_rows = test_4[test_4["scenario"] == "one_shift_baseline"]
    if baseline_rows.empty:
        return

    baseline = baseline_rows.iloc[0]
    scenarios = test_4["scenario"].tolist()
    x_positions = range(len(metrics))
    width = 0.34
    colors = {
        "one_shift_baseline": "#153e7e",
        "two_shift_slots": "#b6003b",
    }

    fig, ax = plt.subplots(figsize=(10, 5.2))
    for scenario_index, scenario in enumerate(scenarios):
        scenario_row = test_4[test_4["scenario"] == scenario].iloc[0]
        values = []
        for metric_col, _ in metrics:
            base_value = baseline.get(metric_col, 0)
            scenario_value = scenario_row.get(metric_col, 0)
            if base_value == 0:
                values.append(0)
            else:
                values.append((scenario_value - base_value) / abs(base_value) * 100)

        offset = (scenario_index - (len(scenarios) - 1) / 2) * width
        ax.bar(
            [position + offset for position in x_positions],
            values,
            width=width,
            label=scenario.replace("_", " "),
            color=colors.get(scenario, "#2f7d32"),
        )

    ax.axhline(0, color="#172033", linewidth=1)
    ax.set_title("Test 4 - Normalised scenario comparison", fontweight="bold")
    ax.set_ylabel("Change relative to one-shift baseline (%)")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels([label for _, label in metrics], rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "test4_normalised_comparison.png", dpi=180)
    plt.close(fig)


def plot_test4_shift_metric_panels(test_4, plots_dir):
    metrics = [
        ("mean_box_fulfilment_rate_pct", "Box fulfilment rate", "Box fulfilment rate (%)"),
        ("mean_postponed_boxes", "Postponed boxes", "Boxes"),
        ("mean_scheduled_economic_value_eur", "Economic value", "EUR"),
        ("mean_setup_time_min", "Setup time", "Minutes"),
        ("mean_operator_utilisation_min", "Operator utilisation", "Operator-minutes"),
        ("mean_avg_capacity_utilisation_shift_2_pct", "Second-shift capacity use", "Utilisation (%)"),
    ]
    scenario_order = ["one_shift_baseline", "two_shift_slots"]
    labels = ["1 shift", "2 shifts"]
    colors = ["#153e7e", "#b6003b"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, (metric_col, title, y_label) in zip(axes.flat, metrics):
        values = []
        for scenario in scenario_order:
            rows = test_4[test_4["scenario"] == scenario]
            values.append(float(rows.iloc[0].get(metric_col, 0)) if not rows.empty else 0)

        bars = ax.bar(labels, values, color=colors, width=0.58)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_ylabel(y_label)
        ax.grid(axis="y", linestyle="--", alpha=0.25)

        baseline_value = values[0]
        two_shift_value = values[1]
        if baseline_value != 0:
            change = (two_shift_value - baseline_value) / abs(baseline_value) * 100
            ax.text(
                bars[1].get_x() + bars[1].get_width() / 2,
                bars[1].get_height(),
                f"{change:+.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    fig.suptitle(
        "Test 4 - Shift structure impact across operational KPIs",
        fontweight="bold",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(plots_dir / "test4_shift_metric_panels.png", dpi=180)
    plt.close(fig)


def plot_test4_daily_capacity_utilisation(test_4_shift, plots_dir):
    daily = (
        test_4_shift.groupby(["scenario", "calendar_day", "shift"])["capacity_utilisation_pct"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5.2))

    baseline = daily[
        (daily["scenario"] == "one_shift_baseline")
        & (daily["shift"] == 1)
    ].sort_values("calendar_day")
    if not baseline.empty:
        ax.plot(
            baseline["calendar_day"],
            baseline["capacity_utilisation_pct"],
            marker="o",
            linewidth=2.2,
            color="#153e7e",
            linestyle="-",
            label="One shift baseline - shift 1",
        )

    two_shift_1 = daily[
        (daily["scenario"] == "two_shift_slots")
        & (daily["shift"] == 1)
    ].sort_values("calendar_day")
    if not two_shift_1.empty:
        ax.plot(
            two_shift_1["calendar_day"],
            two_shift_1["capacity_utilisation_pct"],
            marker="o",
            linewidth=2,
            color="#153e7e",
            linestyle="--",
            label="Two-shift scenario - shift 1",
        )

    two_shift_2 = daily[
        (daily["scenario"] == "two_shift_slots")
        & (daily["shift"] == 2)
    ].sort_values("calendar_day")
    if not two_shift_2.empty:
        ax.plot(
            two_shift_2["calendar_day"],
            two_shift_2["capacity_utilisation_pct"],
            marker="s",
            linewidth=2,
            color="#b6003b",
            linestyle="--",
            label="Two-shift scenario - shift 2",
        )

    ax.set_title("Test 4 - Daily capacity utilisation by shift", fontweight="bold")
    ax.set_xlabel("Planning day")
    ax.set_ylabel("Capacity utilisation (%)")
    ax.set_xticks(sorted(daily["calendar_day"].unique()))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "test4_daily_capacity_utilisation.png", dpi=180)
    plt.close(fig)


def read_optional_csv(path):
    return pd.read_csv(path) if path.exists() else None


def combine_weight_csvs(output_dir, kind):
    combined_path = output_dir / f"test_3_weight_sensitivity_{kind}.csv"
    if combined_path.exists():
        return pd.read_csv(combined_path)

    frames = []
    for weight_index in range(len(WEIGHT_NAMES)):
        path = output_dir / f"test_3_weight_sensitivity_{kind}_weight_{weight_index}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {combined_path}")
    return combined


def add_postponed_box_columns(df):
    """Backfill postponed-box columns in older CSVs without rerunning the GA."""
    if df is None:
        return None

    needs_mean = (
        "mean_postponed_boxes" not in df.columns
        and "mean_box_fulfilment_rate_pct" in df.columns
    )
    needs_raw = (
        "postponed_boxes" not in df.columns
        and "box_fulfilment_rate_pct" in df.columns
    )
    if not needs_mean and not needs_raw:
        return df

    try:
        base_instance = load_july_instance(DEFAULT_INSTANCE_FILE, operators=20)
    except Exception as exc:
        print(f"WARNING: could not backfill postponed boxes: {exc}")
        return df

    base_boxes = total_demand_boxes(base_instance)
    demand_boxes_by_factor = {
        factor: total_demand_boxes(scale_demand(base_instance, factor))
        for factor in DEMAND_FACTORS
    }

    def row_total_boxes(row):
        if "demand_factor" in row and pd.notna(row["demand_factor"]):
            return demand_boxes_by_factor.get(float(row["demand_factor"]), base_boxes)
        return base_boxes

    df = df.copy()
    if needs_mean:
        df["mean_postponed_boxes"] = df.apply(
            lambda row: row_total_boxes(row)
            * (1 - float(row["mean_box_fulfilment_rate_pct"]) / 100),
            axis=1,
        )
    if needs_raw:
        df["postponed_boxes"] = df.apply(
            lambda row: row_total_boxes(row)
            * (1 - float(row["box_fulfilment_rate_pct"]) / 100),
            axis=1,
        )
    return df


def generate_plots(output_dir):
    output_dir = pathlib.Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    test_1 = add_postponed_box_columns(
        read_optional_csv(output_dir / "test_1_demand_scaling_summary.csv")
    )
    if test_1 is not None:
        plot_line(test_1, "demand_factor", "mean_fitness", plots_dir / "test_1_fitness_by_demand.png", "Test 1 - Fitness by demand factor", "Demand factor", "Mean fitness")
        plot_line(test_1, "demand_factor", "mean_postponed_boxes", plots_dir / "test_1_postponed_by_demand.png", "Test 1 - Postponed boxes by demand factor", "Demand factor", "Mean postponed boxes")
        plot_line(test_1, "demand_factor", "mean_box_fulfilment_rate_pct", plots_dir / "test_1_fulfilment_by_demand.png", "Test 1 - Box fulfilment by demand factor", "Demand factor", "Box fulfilment rate (%)")
        plot_dual_axis(test_1, "demand_factor", "mean_scheduled_economic_value_eur", "mean_scheduled_kg", plots_dir / "test_1_value_kg_by_demand.png", "Test 1 - Scheduled value and kg by demand factor", "Demand factor", "Scheduled value (€)", "Scheduled kg")
        plot_dual_axis(test_1, "demand_factor", "mean_setup_time_min", "mean_operator_utilisation_min", plots_dir / "test_1_setup_operator_by_demand.png", "Test 1 - Setup and operator use by demand factor", "Demand factor", "Setup time (min)", "Operator-minutes")

    test_2 = add_postponed_box_columns(
        read_optional_csv(output_dir / "test_2_operator_availability_summary.csv")
    )
    if test_2 is not None:
        plot_line(test_2, "operators", "mean_fitness", plots_dir / "test_2_fitness_by_operators.png", "Test 2 - Fitness by operators", "Operators", "Mean fitness")
        plot_line(test_2, "operators", "mean_postponed_boxes", plots_dir / "test_2_postponed_by_operators.png", "Test 2 - Postponed boxes by operators", "Operators", "Mean postponed boxes")
        plot_line(test_2, "operators", "mean_box_fulfilment_rate_pct", plots_dir / "test_2_fulfilment_by_operators.png", "Test 2 - Box fulfilment by operators", "Operators", "Box fulfilment rate (%)")
        plot_dual_axis(test_2, "operators", "mean_scheduled_economic_value_eur", "mean_scheduled_kg", plots_dir / "test_2_value_kg_by_operators.png", "Test 2 - Scheduled value and kg by operators", "Operators", "Scheduled value (€)", "Scheduled kg")
        plot_dual_axis(test_2, "operators", "mean_setup_time_min", "mean_operator_utilisation_min", plots_dir / "test_2_setup_operator_by_operators.png", "Test 2 - Setup and operator use by operators", "Operators", "Setup time (min)", "Operator-minutes")

    test_3 = add_postponed_box_columns(combine_weight_csvs(output_dir, "summary"))
    if test_3 is not None:
        test_3 = test_3.sort_values(["weight_name", "tested_weight_value"])
        raw_test_3 = add_postponed_box_columns(combine_weight_csvs(output_dir, "raw"))
        plot_test3_weight_profiles(test_3, plots_dir, raw_test_3=raw_test_3)
        plot_test3_fulfilment_heatmap(test_3, plots_dir)
        plot_test3_sensitivity_range(test_3, plots_dir)
        metric_plots = [
            ("mean_fitness", "test_3_fitness_by_weight.png", "Test 3 - Fitness by tested weight", "Mean fitness"),
            ("mean_postponed_boxes", "test_3_postponed_by_weight.png", "Test 3 - Postponed boxes by tested weight", "Mean postponed boxes"),
            ("mean_box_fulfilment_rate_pct", "test_3_fulfilment_by_weight.png", "Test 3 - Box fulfilment by tested weight", "Box fulfilment rate (%)"),
            ("mean_scheduled_economic_value_eur", "test_3_value_by_weight.png", "Test 3 - Scheduled value by tested weight", "Scheduled value (€)"),
            ("mean_setup_time_min", "test_3_setup_by_weight.png", "Test 3 - Setup time by tested weight", "Setup time (min)"),
            ("mean_operator_utilisation_min", "test_3_operator_by_weight.png", "Test 3 - Operator use by tested weight", "Operator-minutes"),
        ]
        for metric, filename, title, y_label in metric_plots:
            plot_line(test_3, "tested_weight_value", metric, plots_dir / filename, title, "Tested weight value", y_label, group_col="weight_name")
        plot_heatmap(test_3, "weight_name", "tested_weight_value", "mean_fitness", plots_dir / "test_3_weight_fitness_heatmap.png", "Test 3 - Fitness heatmap by weight and value", "Mean fitness")
        plot_heatmap(test_3, "weight_name", "tested_weight_value", "mean_postponed_boxes", plots_dir / "test_3_weight_postponed_heatmap.png", "Test 3 - Postponed boxes heatmap by weight and value", "Mean postponed boxes")

    test_4 = add_postponed_box_columns(
        read_optional_csv(output_dir / "test_4_shift_structure_summary.csv")
    )
    if test_4 is not None:
        plot_test4_normalised_comparison(test_4, plots_dir)
        plot_test4_shift_metric_panels(test_4, plots_dir)
        plot_grouped_bars(test_4, "scenario", ["mean_fitness", "mean_postponed_boxes", "mean_box_fulfilment_rate_pct"], plots_dir / "test_4_scenario_comparison.png", "Test 4 - Scenario comparison", "Mean value")
        shift_cols = [
            col
            for col in ["mean_avg_capacity_utilisation_shift_1_pct", "mean_avg_capacity_utilisation_shift_2_pct"]
            if col in test_4.columns
        ]
        if shift_cols:
            plot_grouped_bars(test_4, "scenario", shift_cols, plots_dir / "test_4_shift_capacity_utilisation.png", "Test 4 - Capacity utilisation by shift", "Capacity utilisation (%)")
        operator_cols = [
            col
            for col in ["mean_avg_operators_used_shift_1", "mean_avg_operators_used_shift_2"]
            if col in test_4.columns
        ]
        if operator_cols:
            plot_grouped_bars(test_4, "scenario", operator_cols, plots_dir / "test_4_avg_operators_by_shift.png", "Test 4 - Average operators used by shift", "Average operators")
        peak_cols = [
            col
            for col in ["mean_peak_operators_used_shift_1", "mean_peak_operators_used_shift_2"]
            if col in test_4.columns
        ]
        if peak_cols:
            plot_grouped_bars(test_4, "scenario", peak_cols, plots_dir / "test_4_peak_operators_by_shift.png", "Test 4 - Peak operators used by shift", "Peak operators")

    test_4_shift = read_optional_csv(output_dir / "test_4_shift_structure_per_day_shift_utilisation.csv")
    if test_4_shift is not None:
        plot_test4_daily_capacity_utilisation(test_4_shift, plots_dir)
        two_shift = test_4_shift[test_4_shift["scenario"] == "two_shift_slots"].copy()
        if not two_shift.empty:
            plot_heatmap(two_shift, "calendar_day", "shift", "capacity_utilisation_pct", plots_dir / "test_4_daily_shift_capacity_heatmap.png", "Test 4 - Daily capacity utilisation by shift", "Capacity utilisation (%)")
            plot_heatmap(two_shift, "calendar_day", "shift", "avg_operators_used", plots_dir / "test_4_daily_shift_avg_operators_heatmap.png", "Test 4 - Daily average operators by shift", "Average operators")
            plot_heatmap(two_shift, "calendar_day", "shift", "peak_operators_used", plots_dir / "test_4_daily_shift_peak_operators_heatmap.png", "Test 4 - Daily peak operators by shift", "Peak operators")

    print(f"Wrote plots to {plots_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run final July scenario tests for the GA thesis experiments."
    )
    parser.add_argument(
        "--test",
        choices=["1", "2", "3", "4", "all"],
        default="all",
    )
    parser.add_argument("--instance-file", default=str(DEFAULT_INSTANCE_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Generate PNG plots from existing CSV outputs without running the GA.",
    )
    parser.add_argument(
        "--weight-index",
        type=int,
        choices=range(6),
        default=None,
        help="For Test 3 only: run one weight index, 0..5.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)

    if args.plots_only:
        generate_plots(output_dir)
        return

    base_instance = load_july_instance(pathlib.Path(args.instance_file), operators=20)

    if args.test in ("1", "all"):
        raw, summary = run_test_1(base_instance)
        write_csv(output_dir / "test_1_demand_scaling_raw.csv", raw)
        write_csv(output_dir / "test_1_demand_scaling_summary.csv", summary)

    if args.test in ("2", "all"):
        raw, summary = run_test_2(base_instance)
        write_csv(output_dir / "test_2_operator_availability_raw.csv", raw)
        write_csv(output_dir / "test_2_operator_availability_summary.csv", summary)

    if args.test in ("3", "all"):
        raw, summary = run_test_3(base_instance, weight_index=args.weight_index)
        suffix = (
            f"_weight_{args.weight_index}"
            if args.weight_index is not None
            else ""
        )
        write_csv(output_dir / f"test_3_weight_sensitivity_raw{suffix}.csv", raw)
        write_csv(output_dir / f"test_3_weight_sensitivity_summary{suffix}.csv", summary)

    if args.test in ("4", "all"):
        raw, summary, per_shift = run_test_4(base_instance)
        write_csv(output_dir / "test_4_shift_structure_raw.csv", raw)
        write_csv(output_dir / "test_4_shift_structure_summary.csv", summary)
        write_csv(output_dir / "test_4_shift_structure_per_day_shift_utilisation.csv", per_shift)


if __name__ == "__main__":
    main()
