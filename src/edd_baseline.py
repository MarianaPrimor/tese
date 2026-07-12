import argparse
import csv
import json
from pathlib import Path

from evaluator import (
    compute_max_values,
    create_refs_by_id,
    get_available_line_time_for_day,
    get_capacity_tolerance_for_day,
    get_order_economic_value,
    get_production_time,
    get_setup,
    get_valid_days_for_ref,
    normalised_fitness_breakdown,
    valid_lines_for_ref,
)
from generate_instance import load_real_instance
from geneticalgorithm import (
    enforce_hard_constraints,
    evaluate_with_normalised_fitness,
)


DEFAULT_OPERATIONAL_CONFIG = {
    "shift_start_min": 480,
    "shift_end_min": 990,
    "lunch_break_min": 30,
    "cleaning_time_min": 30,
}


def build_edd_baseline_solution(instance):
    refs_by_id = create_refs_by_id(instance)
    decorated_orders = []

    for order_index, order in enumerate(instance["demand"]):
        ref_id = str(order["ref_id"]).strip()
        ref = refs_by_id.get(ref_id)
        delivery_day = order.get("delivery_date") or instance["n_days"] + 1
        economic_value = get_order_economic_value(order, ref) if ref else 0
        decorated_orders.append((
            delivery_day,
            -economic_value,
            order_index,
            order,
            ref,
            ref_id,
        ))

    decorated_orders.sort()
    solution = []
    occupied_time = {}
    last_family = {}

    for _, _, order_index, order, ref, ref_id in decorated_orders:
        valid_lines = valid_lines_for_ref(ref) if ref else []
        valid_days = get_valid_days_for_ref(instance, ref) if ref else []

        if valid_lines and valid_days:
            delivery_day = order.get("delivery_date")
            on_time_days = [
                day
                for day in valid_days
                if delivery_day is None or day <= delivery_day
            ]
            candidate_days = sorted(on_time_days) + [
                day for day in sorted(valid_days) if day not in set(on_time_days)
            ]
            selected_day = None
            selected_line = None

            for day in candidate_days:
                for line in valid_lines:
                    production_time = get_production_time(
                        ref,
                        line,
                        order["master_boxes"],
                    )

                    if production_time is None:
                        continue

                    key = (day, line)
                    setup_time = get_setup(
                        instance,
                        last_family.get(key),
                        ref["family"],
                    )
                    capacity_limit = (
                        get_available_line_time_for_day(instance, day)
                        + get_capacity_tolerance_for_day(instance, day)
                    )

                    if occupied_time.get(key, 0) + setup_time + production_time <= capacity_limit:
                        selected_day = day
                        selected_line = line
                        occupied_time[key] = (
                            occupied_time.get(key, 0)
                            + setup_time
                            + production_time
                        )
                        last_family[key] = ref["family"]
                        break

                if selected_day is not None:
                    break

            day = selected_day
            line = selected_line
            postponed = selected_day is None
        else:
            day = None
            line = None
            postponed = True

        solution.append({
            "order_id": order_index,
            "ref_id": ref_id,
            "master_boxes": order["master_boxes"],
            "delivery_date": order.get("delivery_date"),
            "delivery_calendar_date": order.get("delivery_calendar_date"),
            "adjusted_delivery_date": order.get("adjusted_delivery_date"),
            "day": day,
            "line": line,
            "postponed": postponed,
        })

    return solution


def metric_row(metrics):
    return {
        "Z": metrics.get("normalised_fitness"),
        "scheduled_kg": metrics.get("scheduled_kg", 0),
        "postponed_kg": metrics.get("postponed_kg", 0),
        "scheduled_economic_value": metrics.get("scheduled_economic_value", 0),
        "postponed_economic_value": metrics.get("postponed_economic_value", 0),
        "setup_total_min": metrics.get("setup_total_min", 0),
        "capacity_utilisation_pct": (
            metrics.get("capacity_utilisation_ratio", 0) * 100
        ),
        "operator_utilisation_pct": (
            metrics.get("operator_usage_minutes", 0)
            / metrics.get("max_values", {}).get("operator_minutes", 1)
            * 100
        ),
        "postponed_orders": metrics.get("postponed_orders", 0),
        "postponed_boxes": metrics.get("postponed_boxes", 0),
        "delay_days_total": metrics.get("delay_days_total", 0),
        "capacity_violations": metrics.get("capacity_violations", 0),
        "operator_violations": metrics.get("operator_violations", 0),
        "infeasible_solution": metrics.get("infeasible_solution", False),
    }


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value


def run_edd_baseline(input_file, output_dir):
    instance = load_real_instance(
        str(input_file),
        operational_config=DEFAULT_OPERATIONAL_CONFIG,
    )
    max_values = compute_max_values(instance)
    initial_solution = build_edd_baseline_solution(instance)
    solution = enforce_hard_constraints(
        initial_solution,
        instance,
        reinsert_postponed=False,
    )
    metrics = evaluate_with_normalised_fitness(solution, instance, max_values)
    metrics["normalised_fitness_breakdown"] = normalised_fitness_breakdown(
        metrics,
        max_values,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    row = metric_row(metrics)

    with (output_dir / "edd_baseline_metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    with (output_dir / "edd_baseline_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(json_safe(metrics), f, indent=2, default=str)

    print("=== EDD BASELINE ===")
    for key, value in row.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    return solution, metrics


def parse_args():
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    parser = argparse.ArgumentParser(description="Deterministic EDD baseline.")
    parser.add_argument(
        "--input",
        default=str(repo_dir / "Inputs_June.xlsx"),
        help="Planning input Excel file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo_dir / "outputs" / "edd_baseline"),
        help="Directory for CSV/JSON outputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_edd_baseline(Path(args.input).resolve(), Path(args.output_dir).resolve())
