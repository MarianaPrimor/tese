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
SETUP_PENALTY = 1


# ============================================================
# AUXILIARY FUNCTIONS
# ============================================================

def _positive_value(value):
    """
    Checks if value is numeric and > 0.
    """
    return isinstance(value, (int, float)) and value > 0


def create_refs_by_id(instance):
    """
    Creates quick access dictionary:
    ref_id -> reference data
    """
    return {ref["id"]: ref for ref in instance["refs"]}


def valid_lines_for_ref(ref):
    """
    Returns valid lines for a reference.
    """
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
    """
    Generates a random production schedule.

    Each order receives:
    - one valid line;
    - one production day;
    - sequence is implicit in list order.
    """
    random.seed(seed)

    refs_by_id = create_refs_by_id(instance)

    solution = []

    for order in instance["demand"]:

        ref_id = str(order["ref_id"]).strip()

if ref_id not in refs_by_id:
    print(f"WARNING: demand ref_id {ref_id} not found in references. Skipping order.")
    continue

ref = refs_by_id[ref_id]

        valid_lines = valid_lines_for_ref(ref)

        if not valid_lines:
            line = None
        else:
            line = random.choice(valid_lines)

        day = random.randint(1, instance["n_days"])

        solution.append({
            "ref_id": order["ref_id"],
            "master_boxes": order["master_boxes"],
            "delivery_date": order["delivery_date"],
            "priority": order["priority"],
            "day": day,
            "line": line,
        })

    return solution


# ============================================================
# OPERATORS
# ============================================================

def calculate_available_operators_per_day(instance):
    """
    Calculates available operators per day.
    """
    operators_per_day = {}

    for d in range(instance["n_days"]):

        day = d + 1

        operators_per_day[day] = sum(
            op["availability"][d]
            for op in instance["operators"]
        )

    return operators_per_day


# ============================================================
# PRODUCTION TIME
# ============================================================

def get_production_time(ref, line, master_boxes):
    """
    Calculates production time of an order.
    """

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
    """
    Calculates required operators.
    """

    if line == "L1":
        return ref["ops_L1_prod"] + ref["ops_L1_finish"]

    if line == "L2":
        return ref["ops_L2_prod"] + ref["ops_L2_finish"]

    return 0


# ============================================================
# SETUPS
# ============================================================

def get_setup(instance, previous_family, current_family):
    """
    Returns setup time between families.
    """

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
    """
    Evaluates a production schedule.

    Lower penalty = better schedule.
    """

    refs_by_id = create_refs_by_id(instance)

    available_operators = calculate_available_operators_per_day(instance)

    production_time_by_day_line = {}
    setup_time_by_day_line = {}
    operators_required_by_day = {}

    total_penalty = 0

    invalid_assignments = 0
    capacity_violations = 0
    delay_days_total = 0
    operator_violations = 0
    total_setup_time = 0

    # sort by day, line and original order
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

        ref = refs_by_id[item["ref_id"]]

        day = item["day"]
        line = item["line"]

        key = (day, line)

        # ----------------------------------------------------
        # INVALID LINE
        # ----------------------------------------------------

        if line not in valid_lines_for_ref(ref):

            total_penalty += INVALID_LINE_PENALTY
            invalid_assignments += 1

            continue

        # ----------------------------------------------------
        # PRODUCTION TIME
        # ----------------------------------------------------

        production_time = get_production_time(
            ref,
            line,
            item["master_boxes"]
        )

        if production_time is None:

            total_penalty += INVALID_LINE_PENALTY
            invalid_assignments += 1

            continue

        # ----------------------------------------------------
        # SETUP
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # OPERATORS
        # ----------------------------------------------------

        required_ops = get_required_operators(ref, line)

        operators_required_by_day[day] = (
            operators_required_by_day.get(day, 0)
            + required_ops
        )

        # ----------------------------------------------------
        # DELAY
        # ----------------------------------------------------

        if day > item["delivery_date"]:

            delay = day - item["delivery_date"]

            delay_days_total += delay

            total_penalty += delay * DELAY_PENALTY

    # ========================================================
    # CAPACITY VIOLATIONS
    # ========================================================

    for key in production_time_by_day_line:

        total_time = (
            production_time_by_day_line[key]
            + setup_time_by_day_line.get(key, 0)
        )

        if total_time > instance["available_line_time_min"]:

            excess = (
                total_time
                - instance["available_line_time_min"]
            )

            capacity_violations += 1

            total_penalty += excess * CAPACITY_PENALTY

    # ========================================================
    # OPERATOR VIOLATIONS
    # ========================================================

    for day, required_ops in operators_required_by_day.items():

        available_ops = available_operators.get(day, 0)

        if required_ops > available_ops:

            excess = required_ops - available_ops

            operator_violations += 1

            total_penalty += excess * OPERATORS_PENALTY

    # ========================================================
    # FINAL SETUP PENALTY
    # ========================================================

    total_penalty += total_setup_time * SETUP_PENALTY

    metrics = {

        "total_penalty": total_penalty,

        "invalid_assignments": invalid_assignments,

        "capacity_violations": capacity_violations,

        "delay_days_total": delay_days_total,

        "operator_violations": operator_violations,

        "setup_total_min": total_setup_time,

        "production_time_by_day_line": production_time_by_day_line,

        "setup_time_by_day_line": setup_time_by_day_line,

        "operators_required_by_day": operators_required_by_day,

        "available_operators": available_operators,
    }

    return metrics


# ============================================================
# PRINT FUNCTIONS
# ============================================================

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

    print(f"Total delay: {metrics['delay_days_total']} days")

    print(f"Operator violations: {metrics['operator_violations']}")

    print(f"Total setup time: {metrics['setup_total_min']:.2f} min")

    print("\nProduction time by day/line:")

    for key, value in metrics["production_time_by_day_line"].items():

        print(f"  {key}: {value:.2f} min")

    print("\nRequired operators by day:")

    for day, value in metrics["operators_required_by_day"].items():

        available = metrics["available_operators"].get(day, 0)

        print(
            f"  Day {day}: "
            f"{value} required / "
            f"{available} available"
        )


# ============================================================
# TEST BLOCK
# ============================================================

if __name__ == "__main__":

    instance = load_real_instance("Inputs_Doceleia.xlsx")

    solution = generate_random_solution(instance, seed=42)

    metrics = evaluate_solution(solution, instance)

    print_solution(solution)

    print_metrics(metrics)