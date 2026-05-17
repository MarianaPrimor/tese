import random

from generate_instance import (
    load_real_instance,
    calculate_production_time
)


# ============================================================
# PENALTY WEIGHTS
# ============================================================

INVALID_LINE_PENALTY = 10000
CAPACITY_PENALTY = 50
DELAY_PENALTY = 100
OPERATORS_PENALTY = 80
SETUP_PENALTY = 0


# ============================================================
# AUXILIARY FUNCTIONS
# ============================================================

def _positive_value(value):
    return isinstance(value, (int, float)) and value > 0


def _normalize_ref_id(ref_id):
    return str(ref_id).strip()


def create_refs_by_id(instance):
    return {
        _normalize_ref_id(ref["id"]): ref
        for ref in instance["refs"]
    }


def valid_lines_for_ref(ref):
    lines = []

    if ref["can_L1"] and _positive_value(ref["rate_L1_prod"]):
        lines.append("L1")

    if ref["can_L2"] and _positive_value(ref["rate_L2_prod"]):
        lines.append("L2")

    return lines


# ============================================================
# RANDOM SOLUTION GENERATION
# ============================================================

def generate_random_solution(instance, seed=42):
    random.seed(seed)

    refs_by_id = create_refs_by_id(instance)
    solution = []

    skipped_orders = 0

    for order_index, order in enumerate(instance["demand"]):
        ref_id = _normalize_ref_id(order["ref_id"])

        if ref_id not in refs_by_id:
            print(
                f"WARNING: demand ref_id {ref_id} not found in references. "
                f"Skipping order."
            )
            skipped_orders += 1
            continue

        ref = refs_by_id[ref_id]
        valid_lines = valid_lines_for_ref(ref)

        if not valid_lines:
            line = None
        else:
            line = random.choice(valid_lines)

        day = random.randint(1, instance["n_days"])

        solution.append({
            "order_id": order_index,
            "ref_id": ref_id,
            "master_boxes": order["master_boxes"],
            "delivery_date": order["delivery_date"],
            "priority": order["priority"],
            "day": day,
            "line": line,
        })

    if skipped_orders > 0:
        print(f"WARNING: skipped {skipped_orders} demand orders.")

    return solution


# ============================================================
# OPERATORS
# ============================================================

def calculate_standard_operators(instance):
    standard_operators = instance.get("standard_operators")

    if standard_operators is None:
        raise ValueError(
            "standard_operators is missing from the instance. "
            "Check generate_instance.py and sheet 1_ESTRUTURA."
        )

    return standard_operators


# ============================================================
# PRODUCTION TIME
# ============================================================

def get_production_time(ref, line, master_boxes):
    if line == "L1":
        rate = ref["rate_L1_prod"]
    elif line == "L2":
        rate = ref["rate_L2_prod"]
    else:
        return None

    return calculate_production_time(
        master_boxes,
        ref["cakes_per_box"],
        rate
    )


# ============================================================
# OPERATORS REQUIRED
# ============================================================

def get_required_operators(ref, line):
    if line == "L1":
        return (ref["ops_L1_prod"] or 0) + (ref["ops_L1_finish"] or 0)

    if line == "L2":
        return (ref["ops_L2_prod"] or 0) + (ref["ops_L2_finish"] or 0)

    return 0


# ============================================================
# SETUPS
# ============================================================

def get_setup(instance, previous_family, current_family):
    if previous_family is None:
        return 0

    matrix = instance["setups_matrix"]

    if (previous_family, current_family) in matrix:
        return matrix[(previous_family, current_family)]

    if previous_family == current_family:
        return 5

    return 30


# ============================================================
# SOLUTION EVALUATION
# ============================================================

def evaluate_solution(solution, instance):
    refs_by_id = create_refs_by_id(instance)
    standard_operators = calculate_standard_operators(instance)

    production_time_by_day_line = {}
    setup_time_by_day_line = {}
    operators_required_by_day_line = {}
    operators_required_by_day = {}
    operator_excess_by_day = {}
    capacity_excess_by_day_line = {}

    total_penalty = 0

    invalid_assignments = 0
    capacity_violations = 0
    delay_days_total = 0
    operator_violations = 0
    total_setup_time = 0
    total_capacity_excess = 0
    total_operator_excess = 0

    sorted_solution = sorted(
        enumerate(solution),
        key=lambda x: (
            x[1]["day"],
            x[1]["line"] or "INVALID",
            x[0]
        )
    )

    last_family_by_day_line = {}

    for _, item in sorted_solution:
        ref_id = _normalize_ref_id(item["ref_id"])

        if ref_id not in refs_by_id:
            total_penalty += INVALID_LINE_PENALTY
            invalid_assignments += 1
            continue

        ref = refs_by_id[ref_id]

        day = item["day"]
        line = item["line"]
        key = (day, line)

        if line not in valid_lines_for_ref(ref):
            total_penalty += INVALID_LINE_PENALTY
            invalid_assignments += 1
            continue

        production_time = get_production_time(
            ref,
            line,
            item["master_boxes"]
        )

        if production_time is None:
            total_penalty += INVALID_LINE_PENALTY
            invalid_assignments += 1
            continue

        previous_family = last_family_by_day_line.get(key)

        setup = get_setup(
            instance,
            previous_family,
            ref["family"]
        )

        last_family_by_day_line[key] = ref["family"]

        production_time_by_day_line[key] = (
            production_time_by_day_line.get(key, 0)
            + production_time
        )

        setup_time_by_day_line[key] = (
            setup_time_by_day_line.get(key, 0)
            + setup
        )

        total_setup_time += setup

        required_ops = get_required_operators(ref, line)

        operators_required_by_day_line[key] = max(
            operators_required_by_day_line.get(key, 0),
            required_ops
        )

        delivery_date = item["delivery_date"]

        if delivery_date is not None and day > delivery_date:
            delay = day - delivery_date
            delay_days_total += delay
            total_penalty += delay * DELAY_PENALTY

    for key in production_time_by_day_line:
        total_time = (
            production_time_by_day_line[key]
        )

        excess = max(0, total_time - instance["available_line_time_min"])
        capacity_excess_by_day_line[key] = excess

        if excess > 0:
            capacity_violations += 1
            total_capacity_excess += excess
            total_penalty += excess * CAPACITY_PENALTY

    for day in range(1, instance["n_days"] + 1):
        required_ops = sum(
            operators_required_by_day_line.get((day, line), 0)
            for line in instance["final_lines"]
        )

        operators_required_by_day[day] = required_ops

        excess = max(0, required_ops - standard_operators)
        operator_excess_by_day[day] = excess

        if excess > 0:
            operator_violations += 1
            total_operator_excess += excess
            total_penalty += excess * OPERATORS_PENALTY

    setup_penalty = total_setup_time * SETUP_PENALTY
    total_penalty += setup_penalty

    metrics = {
        "total_penalty": total_penalty,
        "invalid_assignments": invalid_assignments,
        "capacity_violations": capacity_violations,
        "delay_days_total": delay_days_total,
        "operator_violations": operator_violations,
        "setup_total_min": total_setup_time,
        "total_capacity_excess": total_capacity_excess,
        "total_operator_excess": total_operator_excess,
        "setup_penalty": setup_penalty,
        "production_time_by_day_line": production_time_by_day_line,
        "setup_time_by_day_line": setup_time_by_day_line,
        "capacity_excess_by_day_line": capacity_excess_by_day_line,
        "operators_required_by_day_line": operators_required_by_day_line,
        "operators_required_by_day": operators_required_by_day,
        "operator_excess_by_day": operator_excess_by_day,
        "standard_operators": standard_operators,
    }

    return metrics

def print_solution(solution):
    print("\n=== GENERATED SOLUTION ===")

    for i, item in enumerate(solution, start=1):
        print(
            f"{i:02d}. "
            f"{item['ref_id']} | "
            f"{item['master_boxes']} boxes | "
            f"day {item['day']} | "
            f"{item['line']} | "
            f"delivery {item['delivery_date']} | "
            f"priority {item['priority']}"
        )

def print_metrics(metrics):
    print("\n=== SOLUTION METRICS ===")

    print(f"Total penalty: {metrics['total_penalty']:.2f}")
    print(f"Invalid assignments: {metrics['invalid_assignments']}")
    print(f"Capacity violations: {metrics['capacity_violations']}")
    print(f"Total capacity excess: {metrics['total_capacity_excess']:.2f} min")
    print(f"Total delay: {metrics['delay_days_total']} days")
    print(f"Operator violations: {metrics['operator_violations']}")
    print(f"Total operator excess: {metrics['total_operator_excess']:.2f}")
    print(f"Total setup time: {metrics['setup_total_min']:.2f} min")

    print("\nProduction time by day/line:")

    for key, value in metrics["production_time_by_day_line"].items():
        excess = metrics["capacity_excess_by_day_line"].get(key, 0)

        print(
            f"  {key}: "
            f"{value:.2f} min production / "
            f"{excess:.2f} min excess"
        )

    print("\nRequired operators by day/line:")

    for key, value in metrics["operators_required_by_day_line"].items():
        print(f"  {key}: {value} operators")

    print("\nRequired operators by day:")

    for day, value in metrics["operators_required_by_day"].items():
        excess = metrics["operator_excess_by_day"].get(day, 0)

        print(
            f"  Day {day}: "
            f"{value} required / "
            f"{metrics['standard_operators']} standard / "
            f"{excess} excess"
        )

# ============================================================
# TEST BLOCK
# ============================================================

if __name__ == "__main__":
    instance = load_real_instance("../Inputs_Doceleia.xlsx")

    solution = generate_random_solution(instance, seed=42)

    metrics = evaluate_solution(solution, instance)

    print_solution(solution)

    print_metrics(metrics)
