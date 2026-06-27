import argparse
import copy
import csv
import pathlib
import statistics
import sys

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
    rows = []

    for slot in range(1, instance["n_days"] + 1):
        calendar_day = slot_to_day(slot, shifts_per_day)
        shift = slot_to_shift(slot, shifts_per_day)
        available = get_available_line_time_for_day(instance, slot)

        for line in instance.get("final_lines", []):
            production = production_by_slot_line.get((slot, line), 0)
            rows.append({
                "scenario": scenario,
                "seed": seed,
                "calendar_day": calendar_day,
                "shift": shift,
                "slot": slot,
                "line": line,
                "production_time_min": production,
                "available_time_min": available,
                "utilisation_pct": production / available * 100 if available else 0,
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
                values = [
                    item["utilisation_pct"]
                    for item in shift_rows
                    if item["shift"] == shift
                ]
                row[f"avg_capacity_utilisation_shift_{shift}_pct"] = mean(values)
            rows.append(row)

    return rows, summarise(rows, ["scenario"]), per_shift_rows


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
