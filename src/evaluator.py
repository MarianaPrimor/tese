import math
import os
import random

from generate_instance import (
    load_real_instance,
    calculate_production_time
)

TIME_BUCKET_MIN = 30
FINISHING_DELAY_L1_MIN = 60
SHIFT_START_MIN = 8 * 60

PRODUCTION_LUNCH_START = 12 * 60 + 30
PRODUCTION_LUNCH_END = 13 * 60

FINISHING_LUNCH_START = 13 * 60 + 30
FINISHING_LUNCH_END = 14 * 60


# ============================================================
# PENALTY WEIGHTS
# ============================================================

HOURLY_OPERATORS_PENALTY = 1
DELAY_PENALTY = 1000
CAPACITY_PENALTY = 10000
SETUP_PENALTY = 100
POSTPONEMENT_PENALTY = 10000
ECONOMIC_VALUE_REWARD = 1


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
    fixed_line = ref.get("fixed_line")

    if fixed_line == "L1" and ref["can_L1"] and _positive_value(ref["rate_L1_prod"]):
        return ["L1"]

    if fixed_line == "L2" and ref["can_L2"] and _positive_value(ref["rate_L2_prod"]):
        return ["L2"]

    lines = []

    if ref["can_L1"] and _positive_value(ref["rate_L1_prod"]):
        lines.append("L1")

    if ref["can_L2"] and _positive_value(ref["rate_L2_prod"]):
        lines.append("L2")

    return lines[:1]


def get_valid_days_for_ref(instance, ref):
    monday_days = set(instance.get("monday_days", []))

    if ref.get("monday_forbidden"):
        return [
            day for day in range(1, instance["n_days"] + 1)
            if day not in monday_days
        ]

    return list(range(1, instance["n_days"] + 1))


def round_up_to_bucket(minutes):
    if minutes is None:
        return None

    if minutes <= 0:
        return 0

    return math.ceil(minutes / TIME_BUCKET_MIN) * TIME_BUCKET_MIN


def get_order_kg(item, ref):
    return item.get("master_boxes", 0) * (ref.get("kg_per_master_box") or 0)


def get_order_economic_value(item, ref):
    return item.get("master_boxes", 0) * (ref.get("economic_value_per_master_box") or 0)


def get_postponement_penalty(master_boxes):
    return master_boxes * POSTPONEMENT_PENALTY

# ============================================================
# RANDOM SOLUTION GENERATION
# ============================================================

def generate_random_solution(instance, seed=42, postponement_rate=0.15):
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
        master_boxes = order["master_boxes"]
        valid_lines = valid_lines_for_ref(ref)
        valid_days = get_valid_days_for_ref(instance, ref)

        postponed = bool(valid_lines) and bool(valid_days) and random.random() < postponement_rate

        if postponed:
            line = None
            day = None
        elif not valid_lines or not valid_days:
            line = None
            day = None
            postponed = True
        else:
            line = valid_lines[0]
            day = random.choice(valid_days)

        solution.append({
            "order_id": order_index,
            "ref_id": ref_id,
            "master_boxes": order["master_boxes"],
            "delivery_date": order["delivery_date"],
            "delivery_calendar_date": order.get("delivery_calendar_date"),
            "adjusted_delivery_date": order.get("adjusted_delivery_date"),
            "priority": order["priority"],
            "day": day,
            "line": line,
            "postponed": postponed,
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

    raw_time = calculate_production_time(
        master_boxes,
        ref["cakes_per_box"],
        rate
    )

    return round_up_to_bucket(raw_time)


def get_finishing_time(ref, line, master_boxes):
    production_time = get_production_time(ref, line, master_boxes)

    if production_time is None:
        return None

    if line == "L1":
        return max(0, production_time - FINISHING_DELAY_L1_MIN)

    if line == "L2":
        return 0

    return None


def get_finishing_delay(line):
    if line == "L1":
        return FINISHING_DELAY_L1_MIN

    if line == "L2":
        return 0

    return 0
# ============================================================
# OPERATORS REQUIRED
# ============================================================

def get_required_operators(ref, line):
    if line == "L1":
        return (ref["ops_L1_prod"] or 0) + (ref["ops_L1_finish"] or 0)

    if line == "L2":
        return ref["ops_L2_prod"] or 0

    return 0


def get_production_operators(ref, line):
    if line == "L1":
        return ref["ops_L1_prod"] or 0

    if line == "L2":
        return ref["ops_L2_prod"] or 0

    return 0


def get_finishing_operators(ref, line):
    if line == "L1":
        return ref["ops_L1_finish"] or 0

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

    raise KeyError(
        f"Missing setup time from family '{previous_family}' "
        f"to family '{current_family}' in setups_matrix."
    )

# ============================================================
# TIME SIMULATION
# ============================================================

def schedule_with_lunch(start_time, duration, lunch_start, lunch_end):
    if duration <= 0:
        return start_time, start_time

    if lunch_start <= start_time < lunch_end:
        start_time = lunch_end

    end_time = start_time + duration

    if start_time < lunch_start and end_time > lunch_start:
        end_time += lunch_end - lunch_start

    return start_time, end_time


def simulate_time_schedule(solution, instance):
    refs_by_id = create_refs_by_id(instance)
    operations = []

    sorted_solution = sorted(
        enumerate(solution),
        key=lambda x: (
            x[1].get("day") if x[1].get("day") is not None else instance["n_days"] + 1,
            x[1].get("line") or "POSTPONED",
            x[0]
        )
    )

    groups = {}

    for original_index, item in sorted_solution:
        if item.get("postponed"):
            continue

        key = (item["day"], item["line"])
        groups.setdefault(key, []).append((original_index, item))

    for (day, line), sequence in groups.items():
        if line not in instance["final_lines"]:
            continue

        current_time = SHIFT_START_MIN
        previous_family = None
        previous_prod_ops = 0

        for original_index, item in sequence:
            ref_id = str(item["ref_id"]).strip()

            if ref_id not in refs_by_id:
                continue

            ref = refs_by_id[ref_id]
            master_boxes = item.get("master_boxes", 0)

            if previous_family is not None:
                setup_time = get_setup(instance, previous_family, ref["family"])
                setup_start = current_time
                setup_end = setup_start + setup_time

                operations.append({
                    "day": day,
                    "line": line,
                    "ref_id": item["ref_id"],
                    "master_boxes": master_boxes,
                    "operation": "setup",
                    "start": setup_start,
                    "end": setup_end,
                    "operators": previous_prod_ops,
                })

                current_time = setup_end

            production_time = get_production_time(ref, line, master_boxes) or 0
            prod_start, prod_end = schedule_with_lunch(
                current_time,
                production_time,
                PRODUCTION_LUNCH_START,
                PRODUCTION_LUNCH_END
            )

            prod_ops = get_production_operators(ref, line)

            operations.append({
                "day": day,
                "line": line,
                "ref_id": item["ref_id"],
                "master_boxes": master_boxes,
                "operation": "production",
                "start": prod_start,
                "end": prod_end,
                "operators": prod_ops,
            })

            if line == "L1" and production_time > FINISHING_DELAY_L1_MIN:
                finish_start = prod_start + FINISHING_DELAY_L1_MIN
                finish_end = prod_end
                finish_ops = get_finishing_operators(ref, line)

                operations.append({
                    "day": day,
                    "line": line,
                    "ref_id": item["ref_id"],
                    "master_boxes": master_boxes,
                    "operation": "finishing",
                    "start": finish_start,
                    "end": finish_end,
                    "operators": finish_ops,
                })

            current_time = prod_end
            previous_family = ref["family"]
            previous_prod_ops = prod_ops

    return operations


def calculate_operator_usage_by_time(operations, standard_operators):
    usage = {}

    for op in operations:
        if op["end"] <= op["start"]:
            continue

        start_bucket = int(op["start"] // TIME_BUCKET_MIN)
        end_bucket = int((op["end"] - 1) // TIME_BUCKET_MIN)

        for bucket in range(start_bucket, end_bucket + 1):
            key = (op["day"], bucket)
            usage[key] = usage.get(key, 0) + op["operators"]

    excess = {
        key: max(0, value - standard_operators)
        for key, value in usage.items()
    }

    return usage, excess

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
    infeasible_solution = False

    invalid_assignments = 0
    capacity_violations = 0
    delay_days_total = 0
    operator_violations = 0
    total_setup_time = 0
    total_capacity_excess = 0
    total_operator_excess = 0
    postponed_orders = 0
    postponement_penalty = 0
    postponed_by_due_date = {}
    scheduled_kg = 0
    postponed_kg = 0
    scheduled_economic_value = 0
    postponed_economic_value = 0
    monday_violations = 0

    sorted_solution = sorted(
        enumerate(solution),
        key=lambda x: (
            x[1].get("day") if x[1].get("day") is not None else instance["n_days"] + 1,
            x[1].get("line") or "POSTPONED",
            x[0]
        )
    )

    last_family_by_day_line = {}

    for _, item in sorted_solution:
        ref_id = _normalize_ref_id(item["ref_id"])

        if ref_id not in refs_by_id:
            invalid_assignments += 1
            infeasible_solution = True
            continue

        ref = refs_by_id[ref_id]

        if item.get("postponed"):
            delivery_date = item.get("delivery_date")
            postponed_kg += get_order_kg(item, ref)
            postponed_economic_value += get_order_economic_value(item, ref)
            penalty = get_postponement_penalty(
                item["master_boxes"]
            )
            total_penalty += penalty
            postponement_penalty += penalty
            postponed_orders += 1
            postponed_by_due_date[delivery_date] = (
                postponed_by_due_date.get(delivery_date, 0) + 1
            )
            continue

        day = item["day"]
        line = item["line"]
        key = (day, line)

        if line not in valid_lines_for_ref(ref):
            invalid_assignments += 1
            infeasible_solution = True
            continue

        if ref.get("monday_forbidden") and day in set(instance.get("monday_days", [])):
            invalid_assignments += 1
            monday_violations += 1
            infeasible_solution = True
            continue

        production_time = get_production_time(
            ref,
            line,
            item["master_boxes"]
        )

        if production_time is None:
            invalid_assignments += 1
            infeasible_solution = True
            continue

        scheduled_kg += get_order_kg(item, ref)
        scheduled_economic_value += get_order_economic_value(item, ref)

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
            + setup_time_by_day_line.get(key, 0)
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

    setup_penalty = total_setup_time * SETUP_PENALTY
    total_penalty += setup_penalty
    time_operations = simulate_time_schedule(solution, instance)

    operators_required_by_time, operator_excess_by_time = (
        calculate_operator_usage_by_time(
            time_operations,
            standard_operators
        )
    )

    total_operator_excess_by_time = sum(
        operator_excess_by_time.values()
    )

    peak_operators = max(
        operators_required_by_time.values(),
        default=0
    )

    hourly_operator_penalty = (
        total_operator_excess_by_time * HOURLY_OPERATORS_PENALTY
    )

    total_penalty += hourly_operator_penalty


    economic_value_reward = scheduled_economic_value * ECONOMIC_VALUE_REWARD
    total_penalty -= economic_value_reward

    if infeasible_solution:
        total_penalty = float("inf")

    metrics = {
        "total_penalty": total_penalty,
        "invalid_assignments": invalid_assignments,
        "infeasible_solution": infeasible_solution,
        "capacity_violations": capacity_violations,
        "delay_days_total": delay_days_total,
        "operator_violations": operator_violations,
        "setup_total_min": total_setup_time,
        "total_capacity_excess": total_capacity_excess,
        "total_operator_excess": total_operator_excess,
        "postponed_orders": postponed_orders,
        "postponement_penalty": postponement_penalty,
        "postponed_by_due_date": postponed_by_due_date,
        "setup_penalty": setup_penalty,
        "total_operator_excess_by_time": total_operator_excess_by_time,
        "total_hourly_operator_excess": total_operator_excess_by_time,
        "production_time_by_day_line": production_time_by_day_line,
        "setup_time_by_day_line": setup_time_by_day_line,
        "capacity_excess_by_day_line": capacity_excess_by_day_line,
        "operators_required_by_day_line": operators_required_by_day_line,
        "operators_required_by_day": operators_required_by_day,
        "operator_excess_by_day": operator_excess_by_day,
        "standard_operators": standard_operators,
        "time_operations": time_operations,
        "operators_required_by_time": operators_required_by_time,
        "operator_excess_by_time": operator_excess_by_time,
        "peak_operators": peak_operators,
        "hourly_operator_penalty": hourly_operator_penalty,
        "scheduled_kg": scheduled_kg,
        "postponed_kg": postponed_kg,
        "scheduled_economic_value": scheduled_economic_value,
        "postponed_economic_value": postponed_economic_value,
        "economic_value_reward": economic_value_reward,
        "monday_violations": monday_violations,
    }

    return metrics

def print_solution(solution):
    print("\n=== GENERATED SOLUTION ===")

    for i, item in enumerate(solution, start=1):
        status = "POSTPONED" if item.get("postponed") else f"day {item['day']} | {item['line']}"

        print(
            f"{i:02d}. "
            f"{item['ref_id']} | "
            f"{item['master_boxes']} boxes | "
            f"{status} | "
            f"delivery {item['delivery_date']} | "
            f"priority {item['priority']}"
        )

def print_metrics(metrics):
    print("\n=== SOLUTION METRICS ===")

    if "computation_time_sec" in metrics:
        print(f"Computation time: {metrics['computation_time_sec']:.2f} sec")

    print(f"Total penalty: {metrics['total_penalty']:.2f}")
    print(f"Total economic reward: {metrics.get('economic_value_reward', 0):.2f}")
    print(f"Scheduled economic value: {metrics.get('scheduled_economic_value', 0):.2f}")
    print(f"Postponed economic value: {metrics.get('postponed_economic_value', 0):.2f}")
    print(f"Total capacity excess: {metrics['total_capacity_excess']:.2f} min")
    print(f"Total setup time: {metrics['setup_total_min']:.2f} min")
    print(f"Total delay: {metrics['delay_days_total']} days")
    print(f"Postponed orders: {metrics['postponed_orders']}")
    print(f"Invalid assignments: {metrics['invalid_assignments']}")
    print(f"Capacity violations: {metrics['capacity_violations']}")
    print(f"Operator violations: {metrics['operator_violations']}")
    print(f"Total operator excess: {metrics['total_operator_excess']:.2f}")
    print(f"Peak operators by time: {metrics['peak_operators']}")
    print(f"Total hourly operator excess: {metrics['total_operator_excess_by_time']:.2f}")
    print(f"Postponement penalty: {metrics['postponement_penalty']:.2f}")
    print(f"Hourly operator penalty: {metrics['hourly_operator_penalty']:.2f}")
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


def _format_minutes(value):
    value = int(round(value))
    hours = value // 60
    minutes = value % 60
    return f"{hours:02d}:{minutes:02d}"


def print_validation_report(solution, instance, title="VALIDATION REPORT"):
    refs_by_id = create_refs_by_id(instance)
    metrics = evaluate_solution(solution, instance)

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print("\n1. ASSIGNMENT: EACH ORDER SCHEDULED ONCE OR POSTPONED")
    seen_order_ids = {}

    for item in solution:
        order_id = item.get("order_id")
        seen_order_ids[order_id] = seen_order_ids.get(order_id, 0) + 1

    for order_id in range(len(instance["demand"])):
        count = seen_order_ids.get(order_id, 0)
        status = "OK" if count == 1 else "ERROR"
        print(f"  Order {order_id:02d}: occurrences in solution = {count} -> {status}")

    print("\n2. LINE COMPATIBILITY AND MONDAY RESTRICTION")

    for item in sorted(solution, key=lambda x: x.get("order_id", 0)):
        ref_id = str(item["ref_id"]).strip()
        ref = refs_by_id.get(ref_id)

        if ref is None:
            print(f"  Order {item.get('order_id')}: {ref_id} -> ERROR reference not found")
            continue

        if item.get("postponed"):
            print(f"  Order {item.get('order_id'):02d}: {ref_id} -> POSTPONED")
            continue

        valid_lines = valid_lines_for_ref(ref)
        valid_days = get_valid_days_for_ref(instance, ref)
        line_ok = item["line"] in valid_lines
        day_ok = item["day"] in valid_days
        monday_text = (
            "monday forbidden"
            if ref.get("monday_forbidden")
            else "monday allowed"
        )

        print(
            f"  Order {item.get('order_id'):02d}: {ref_id} | "
            f"line {item['line']} valid {valid_lines} -> {'OK' if line_ok else 'ERROR'} | "
            f"day {item['day']} valid {valid_days} ({monday_text}) -> {'OK' if day_ok else 'ERROR'}"
        )

    print("\n3. DELAY CALCULATION")

    for item in sorted(solution, key=lambda x: x.get("order_id", 0)):
        if item.get("postponed"):
            print(f"  Order {item.get('order_id'):02d}: POSTPONED -> delay not calculated")
            continue

        delay = max(0, item["day"] - item["delivery_date"])
        print(
            f"  Order {item.get('order_id'):02d}: production day {item['day']} - "
            f"delivery day {item['delivery_date']} = {delay} delay days"
        )

    print("\n4. SETUP TRANSITIONS BY DAY AND LINE")
    sorted_solution = sorted(
        enumerate(solution),
        key=lambda x: (
            x[1].get("day") if x[1].get("day") is not None else instance["n_days"] + 1,
            x[1].get("line") or "POSTPONED",
            x[0],
        )
    )
    last_family_by_day_line = {}
    setup_total = 0

    for _, item in sorted_solution:
        if item.get("postponed"):
            continue

        ref = refs_by_id[str(item["ref_id"]).strip()]
        key = (item["day"], item["line"])
        previous_family = last_family_by_day_line.get(key)
        current_family = ref["family"]
        setup = get_setup(instance, previous_family, current_family)
        setup_total += setup

        if previous_family is None:
            transition = f"START -> {current_family}"
        else:
            transition = f"{previous_family} -> {current_family}"

        print(
            f"  Day {item['day']} {item['line']}: "
            f"{transition} before {item['ref_id']} = {setup:.2f} min"
        )

        last_family_by_day_line[key] = current_family

    print(f"  Total setup recalculated: {setup_total:.2f} min")
    print(f"  Total setup from evaluator: {metrics['setup_total_min']:.2f} min")

    print("\n5. DAILY LINE CAPACITY")

    keys = set(metrics["production_time_by_day_line"].keys())
    keys.update(metrics["setup_time_by_day_line"].keys())

    for key in sorted(keys):
        production_time = metrics["production_time_by_day_line"].get(key, 0)
        setup_time = metrics["setup_time_by_day_line"].get(key, 0)
        total_time = production_time + setup_time
        excess = metrics["capacity_excess_by_day_line"].get(key, 0)
        available = instance["available_line_time_min"]
        status = "OK" if excess == 0 else "PENALIZED"

        print(
            f"  Day {key[0]} {key[1]}: production {production_time:.2f} + "
            f"setup {setup_time:.2f} = {total_time:.2f} / "
            f"{available:.2f} min | excess {excess:.2f} -> {status}"
        )

    print("\n6. OPERATOR DEMAND BY TIME PERIOD")

    for key, required in sorted(metrics["operators_required_by_time"].items()):
        day, bucket = key
        start = bucket * TIME_BUCKET_MIN
        end = start + TIME_BUCKET_MIN
        excess = metrics["operator_excess_by_time"].get(key, 0)
        status = "OK" if excess == 0 else "PENALIZED"

        print(
            f"  Day {day} {_format_minutes(start)}-{_format_minutes(end)}: "
            f"{required} required / {metrics['standard_operators']} standard | "
            f"excess {excess} -> {status}"
        )

    print("\n7. SAME LINE PRODUCTION OVERLAP CHECK")
    operations = metrics.get("time_operations", [])
    production_operations = [
        op for op in operations
        if op["operation"] in ["setup", "production"]
    ]
    overlap_count = 0

    for i, op_a in enumerate(production_operations):
        for op_b in production_operations[i + 1:]:
            same_resource = (
                op_a["day"] == op_b["day"]
                and op_a["line"] == op_b["line"]
            )
            overlaps = op_a["start"] < op_b["end"] and op_b["start"] < op_a["end"]

            if same_resource and overlaps:
                overlap_count += 1
                print(
                    f"  OVERLAP day {op_a['day']} {op_a['line']}: "
                    f"{op_a['ref_id']} {op_a['operation']} "
                    f"{_format_minutes(op_a['start'])}-{_format_minutes(op_a['end'])} "
                    f"with {op_b['ref_id']} {op_b['operation']} "
                    f"{_format_minutes(op_b['start'])}-{_format_minutes(op_b['end'])}"
                )

    if overlap_count == 0:
        print("  No setup/production overlaps found on the same day and line -> OK")
    else:
        print(f"  Total overlaps found: {overlap_count} -> ERROR")

    return metrics

# ============================================================
# TEST BLOCK
# ============================================================

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    instance_path = os.path.abspath(
        os.path.join(script_dir, "..", "Inputs_Doceleia.xlsx")
    )

    instance = load_real_instance(instance_path)

    solution = generate_random_solution(instance, seed=42)

    metrics = evaluate_solution(solution, instance)

    print_solution(solution)

    print_metrics(metrics)

