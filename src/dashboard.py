import os
from copy import deepcopy
from datetime import date, time, timedelta

import altair as alt
import pandas as pd
import streamlit as st

from generate_instance import load_real_instance
from geneticalgorithm import run_genetic_algorithm
from evaluator import (
    create_refs_by_id,
    compute_max_values,
    evaluate_solution,
    get_available_line_time_for_day,
    get_capacity_tolerance_for_day,
    get_production_time,
    get_standard_operators_for_day,
    get_setup,
    get_valid_days_for_ref,
    TIME_BUCKET_MIN,
    valid_lines_for_ref,
    normalised_fitness,
    normalised_fitness_breakdown,
)


LINE_COLORS = ["#153e7e", "#b6003b"]
OVERLOAD_BG = "#f8d7da"
OVERLOAD_TEXT = "#7f1d1d"
NEUTRAL_BG = "#ffffff"
NEAR_LIMIT_BG = "#fff3cd"
OK_BG = "#d1e7dd"
OK_TEXT = "#0f5132"

POPULATION_SIZE = 100
GENERATIONS = 100
MUTATION_RATE = 0.10
ELITE_SIZE = 5
TOURNAMENT_SIZE = 3
RANDOM_SEED = 42
LUNCH_BREAK_MIN = 30


def format_date(value):
    if value is None:
        return ""

    if hasattr(value, "strftime"):
        return value.strftime("%d/%m/%Y")

    return str(value)


def format_time_from_minutes(value):
    if value is None:
        return ""

    value = int(round(value))
    if value == 24 * 60:
        return "24:00"

    day_offset = value // (24 * 60)
    value = value % (24 * 60)
    hours = value // 60
    minutes = value % 60

    if day_offset > 0:
        return f"+{day_offset}d {hours:02d}:{minutes:02d}"

    return f"{hours:02d}:{minutes:02d}"


def minutes_from_time(value):
    return value.hour * 60 + value.minute


def build_working_days_from_dashboard(start_date, end_date, non_working_days):
    working_days = []
    current = start_date
    non_working_days = set(non_working_days or [])

    while current <= end_date:
        if current.weekday() < 5 and current not in non_working_days:
            working_days.append(current)

        current += timedelta(days=1)

    return working_days


def apply_dashboard_overrides(
    instance,
    working_days,
    operators_by_day,
    start_times_by_day,
    end_times_by_day,
    shifts_by_day,
):
    daily_capacity_min = {}
    daily_shift_start_min = {}
    daily_shift_end_min = {}
    daily_shifts = {}

    cleaning_time = instance.get("end_of_day_cleaning_time_min", 0)

    for day in range(1, len(working_days) + 1):
        n_shifts = shifts_by_day.get(day, 1)
        start_min = minutes_from_time(start_times_by_day[day])
        end_min = minutes_from_time(end_times_by_day[day])

        if n_shifts == 2:
            gross_capacity = 17 * 60
        else:
            gross_capacity = max(0, end_min - start_min)

        available_capacity = max(
            0,
            gross_capacity
            - LUNCH_BREAK_MIN * n_shifts
            - cleaning_time * n_shifts
        )

        daily_shift_start_min[day] = start_min
        daily_shift_end_min[day] = start_min + gross_capacity
        daily_capacity_min[day] = available_capacity
        daily_shifts[day] = n_shifts

    monday_days = [
        index + 1
        for index, working_day in enumerate(working_days)
        if working_day.weekday() == 0
    ]

    instance["working_days"] = working_days
    instance["n_days"] = len(working_days)
    instance["days"] = [f"day_{i + 1}" for i in range(len(working_days))]
    instance["monday_days"] = monday_days
    instance["standard_operators_by_day"] = operators_by_day
    instance["daily_shift_start_min"] = daily_shift_start_min
    instance["daily_shift_end_min"] = daily_shift_end_min
    instance["daily_capacity_min"] = daily_capacity_min
    instance["daily_shifts"] = daily_shifts

    if daily_capacity_min:
        instance["available_line_time_min"] = min(daily_capacity_min.values())

    return instance


def get_production_date(instance, day):
    if day is None:
        return "Postponed"

    working_days = instance.get("working_days", [])

    if working_days and 1 <= day <= len(working_days):
        return format_date(working_days[day - 1])

    return f"Day {day}"


def get_planning_month(instance):
    working_days = instance.get("working_days", [])
    month_names = {
        1: "janeiro",
        2: "fevereiro",
        3: "março",
        4: "abril",
        5: "maio",
        6: "junho",
        7: "julho",
        8: "agosto",
        9: "setembro",
        10: "outubro",
        11: "novembro",
        12: "dezembro",
    }

    if working_days:
        first_day = working_days[0]
        return f"{month_names[first_day.month]} {first_day.year}"

    return "horizonte de planeamento"


def build_instance_signature(instance):
    demand_signature = tuple(
        (
            index,
            str(order.get("ref_id", "")).strip(),
            order.get("master_boxes", 0),
            order.get("delivery_date"),
        )
        for index, order in enumerate(instance.get("demand", []))
    )
    reference_signature = tuple(
        sorted(
            (
                str(ref.get("id", "")).strip(),
                ref.get("economic_value_per_master_box", 0),
            )
            for ref in instance.get("refs", [])
        )
    )
    operating_signature = (
        "normalised_objective_v3",
        tuple(instance.get("working_days", [])),
        tuple(sorted(instance.get("daily_capacity_min", {}).items())),
        tuple(sorted(instance.get("standard_operators_by_day", {}).items())),
    )

    return demand_signature, reference_signature, operating_signature


def validate_solution_orders(solution, instance):
    expected_ids = set(range(len(instance.get("demand", []))))
    solution_ids = [item.get("order_id") for item in solution]
    actual_ids = set(solution_ids)
    duplicates = sorted(
        order_id
        for order_id in actual_ids
        if solution_ids.count(order_id) > 1
    )
    missing = sorted(expected_ids - actual_ids)
    unexpected = sorted(actual_ids - expected_ids)

    return duplicates, missing, unexpected


def get_delivery_date(instance, solution_item):
    delivery_date = solution_item.get("delivery_calendar_date")

    if delivery_date is not None:
        return format_date(delivery_date)

    order_id = solution_item.get("order_id")

    if order_id is not None and order_id < len(instance["demand"]):
        demand_order = instance["demand"][order_id]
        delivery_date = demand_order.get("delivery_calendar_date")

        if delivery_date is not None:
            return format_date(delivery_date)

    return str(solution_item.get("delivery_date", ""))


def solution_sort_key(instance, item_with_index):
    original_index, item = item_with_index
    day = item.get("day")

    return (
        day if day is not None else instance["n_days"] + 1,
        item.get("line") or "POSTPONED",
        original_index,
    )


def build_plan_df(instance, best_solution):
    refs_by_id = create_refs_by_id(instance)
    plan_rows = []
    last_family_by_day_line = {}
    sequence_by_day_line = {}

    sorted_solution = sorted(
        enumerate(best_solution),
        key=lambda x: solution_sort_key(instance, x),
    )

    for original_index, item in sorted_solution:
        status = "Postponed" if item.get("postponed") else "Scheduled"
        day = item.get("day")
        line = item.get("line") or "POSTPONED"
        ref_id = str(item.get("ref_id")).strip()
        ref = refs_by_id.get(ref_id)
        key = (day, line)

        sequence_by_day_line[key] = sequence_by_day_line.get(key, 0) + 1
        sequence = sequence_by_day_line[key]
        production_time = 0
        setup_time = 0
        family = ""
        kg = 0
        economic_value = 0

        if ref is not None:
            kg = item.get("master_boxes", 0) * (ref.get("kg_per_master_box") or 0)
            economic_value = item.get("master_boxes", 0) * (ref.get("economic_value_per_master_box") or 0)

        if status == "Scheduled" and ref is not None and line in ["L1", "L2"]:
            production_time = get_production_time(
                ref,
                line,
                item.get("master_boxes", 0),
            ) or 0
            previous_family = last_family_by_day_line.get(key)
            setup_time = get_setup(instance, previous_family, ref["family"])
            last_family_by_day_line[key] = ref["family"]
            family = ref["family"]

        plan_rows.append({
            "Status": status,
            "Production date": get_production_date(instance, day),
            "Day": day,
            "Line": line,
            "Seq.": sequence if status == "Scheduled" else "",
            "Reference": ref_id,
            "Family": family,
            "Master boxes": item.get("master_boxes", 0),
            "Kg": round(kg, 2),
            "Economic value": round(economic_value, 2),
            "Production time (min)": round(production_time, 1),
            "Setup time (min)": round(setup_time, 1),
            "Delivery date": get_delivery_date(instance, item),
            "Delivery day": item.get("delivery_date", ""),
        })

    return pd.DataFrame(plan_rows)


def build_scenario_editor_df(instance, solution):
    refs_by_id = create_refs_by_id(instance)
    rows = []
    sequence_by_day_line = {}

    for item in solution:
        postponed = bool(item.get("postponed"))
        day = item.get("day")
        line = item.get("line")
        ref_id = str(item.get("ref_id", "")).strip()
        ref = refs_by_id.get(ref_id, {})
        key = (day, line)

        if postponed:
            sequence = None
        else:
            sequence_by_day_line[key] = sequence_by_day_line.get(key, 0) + 1
            sequence = sequence_by_day_line[key]

        master_boxes = item.get("master_boxes", 0) or 0
        economic_value = (
            master_boxes
            * (ref.get("economic_value_per_master_box") or 0)
        )

        rows.append({
            "ID da ordem": item.get("order_id"),
            "Referência": ref_id,
            "Caixas master": master_boxes,
            "Linha": line or "",
            "Dia": day,
            "Sequência": sequence,
            "Adiado": postponed,
            "Dia de entrega": item.get("delivery_date"),
            "Valor económico": round(economic_value, 2),
        })

    return pd.DataFrame(rows)


def build_solution_from_scenario(instance, base_solution, edited_df):
    refs_by_id = create_refs_by_id(instance)
    genes_by_order_id = {
        item.get("order_id"): deepcopy(item)
        for item in base_solution
    }
    ordered_genes = []
    errors = []

    for row_index, row in edited_df.iterrows():
        order_id = row.get("ID da ordem")
        gene = genes_by_order_id.get(order_id)

        if gene is None:
            errors.append(f"Linha {row_index + 1}: ordem {order_id} não encontrada.")
            continue

        ref_id = str(gene.get("ref_id", "")).strip()
        ref = refs_by_id.get(ref_id)

        if ref is None:
            errors.append(
                f"Ordem {order_id}: referência {ref_id} não encontrada."
            )
            continue

        postponed = bool(row.get("Adiado"))
        sequence_value = row.get("Sequência")
        sequence = (
            int(sequence_value)
            if pd.notna(sequence_value)
            else len(base_solution) + row_index
        )

        if postponed:
            gene["day"] = None
            gene["line"] = None
            gene["postponed"] = True
            sort_key = (
                instance["n_days"] + 1,
                "POSTPONED",
                sequence,
                order_id,
            )
        else:
            day_value = row.get("Dia")

            if pd.isna(day_value):
                errors.append(f"Ordem {order_id}: escolha um dia de produção.")
                continue

            day = int(day_value)
            valid_days = get_valid_days_for_ref(instance, ref)
            valid_lines = valid_lines_for_ref(ref)

            if day not in valid_days:
                errors.append(
                    f"Ordem {order_id} ({ref_id}): o dia {day} não é permitido."
                )
                continue

            if not valid_lines:
                errors.append(
                    f"Ordem {order_id} ({ref_id}): não existe linha válida."
                )
                continue

            gene["day"] = day
            gene["line"] = valid_lines[0]
            gene["postponed"] = False
            sort_key = (day, gene["line"], sequence, order_id)

        ordered_genes.append((sort_key, gene))

    ordered_genes.sort(key=lambda item: item[0])
    return [gene for _, gene in ordered_genes], errors


def build_scenario_comparison_df(ga_metrics, manual_metrics):
    comparison_rows = [
        (
            "Penalização total",
            ga_metrics.get("total_penalty", 0),
            manual_metrics.get("total_penalty", 0),
            "min",
        ),
        (
            "Excesso de capacidade (min)",
            ga_metrics.get("total_capacity_excess", 0),
            manual_metrics.get("total_capacity_excess", 0),
            "min",
        ),
        (
            "Tempo de setup (min)",
            ga_metrics.get("setup_total_min", 0),
            manual_metrics.get("setup_total_min", 0),
            "min",
        ),
        (
            "Dias de atraso",
            ga_metrics.get("delay_days_total", 0),
            manual_metrics.get("delay_days_total", 0),
            "min",
        ),
        (
            "Ordens adiadas",
            ga_metrics.get("postponed_orders", 0),
            manual_metrics.get("postponed_orders", 0),
            "min",
        ),
        (
            "Utilização dos operadores (operador-min)",
            ga_metrics.get("operator_usage_minutes", 0),
            manual_metrics.get("operator_usage_minutes", 0),
            "max",
        ),
        (
            "Valor económico planeado",
            ga_metrics.get("scheduled_economic_value", 0),
            manual_metrics.get("scheduled_economic_value", 0),
            "max",
        ),
    ]
    rows = []

    for indicator, ga_value, manual_value, direction in comparison_rows:
        difference = manual_value - ga_value
        improved = difference < 0 if direction == "min" else difference > 0

        if abs(difference) < 1e-9:
            result = "Sem alteração"
        elif improved:
            result = "Melhor"
        else:
            result = "Pior"

        rows.append({
            "Indicador": indicator,
            "Solução GA": round(ga_value, 2),
            "Cenário manual": round(manual_value, 2),
            "Diferença": round(difference, 2),
            "Resultado": result,
        })

    return pd.DataFrame(rows)


def build_capacity_df(instance, best_metrics):
    rows = []
    keys = set(best_metrics["production_time_by_day_line"].keys())
    keys.update(best_metrics["setup_time_by_day_line"].keys())

    for day, line in sorted(keys):
        production_time = best_metrics["production_time_by_day_line"].get((day, line), 0)
        setup_time = best_metrics["setup_time_by_day_line"].get((day, line), 0)
        occupied_time = production_time + setup_time
        available_time = get_available_line_time_for_day(instance, day)
        excess = max(0, occupied_time - available_time)
        utilization = occupied_time / available_time * 100 if available_time else 0

        rows.append({
            "Production date": get_production_date(instance, day),
            "Day": day,
            "Line": line,
            "Shifts": instance.get("daily_shifts", {}).get(day, 1),
            "Production time (min)": round(production_time, 1),
            "Setup time (min)": round(setup_time, 1),
            "Occupied time (min)": round(occupied_time, 1),
            "Available time (min)": round(available_time, 1),
            "Capacity excess (min)": round(excess, 1),
            "Utilization (%)": round(utilization, 1),
        })

    return pd.DataFrame(rows)


def build_compact_schedule_df(instance, plan_df, best_metrics):
    rows = []

    for day in range(1, instance["n_days"] + 1):
        day_df = plan_df[(plan_df["Day"] == day) & (plan_df["Status"] == "Scheduled")]
        row = {
            "Production date": get_production_date(instance, day),
            "Day": day,
            "Shifts": instance.get("daily_shifts", {}).get(day, 1),
        }

        max_utilization = 0
        max_excess = 0

        for line in instance["final_lines"]:
            line_df = day_df[day_df["Line"] == line].sort_values("Seq.")
            refs = " -> ".join(
                f"{row['Reference']} ({row['Master boxes']} boxes)"
                for _, row in line_df.iterrows()
            )
            production_time = best_metrics["production_time_by_day_line"].get((day, line), 0)
            setup_time = best_metrics["setup_time_by_day_line"].get((day, line), 0)
            available = get_available_line_time_for_day(instance, day)
            occupied = production_time + setup_time
            excess = max(0, occupied - available)
            utilization = occupied / available * 100 if available else 0

            row[f"{line} sequence"] = refs
            row[f"{line} production"] = f"{production_time:.1f} min"
            row[f"{line} setup"] = f"{setup_time:.1f} min"
            row[f"{line} excess"] = f"{excess:.1f} min"

            max_utilization = max(max_utilization, utilization)
            max_excess = max(max_excess, excess)

        if max_excess > get_capacity_tolerance_for_day(instance, day):
            row["Status"] = "Overloaded"
        elif max_utilization >= 90:
            row["Status"] = "Near limit"
        else:
            row["Status"] = "OK"

        rows.append(row)

    return pd.DataFrame(rows)


def add_normalised_fitness_metrics(metrics, max_values):
    raw_total_penalty = metrics.get("total_penalty", 0)
    breakdown = normalised_fitness_breakdown(metrics, max_values)
    score = normalised_fitness(metrics, max_values)

    metrics["raw_total_penalty"] = raw_total_penalty
    metrics["normalised_fitness"] = score
    metrics["normalised_fitness_breakdown"] = breakdown
    metrics["total_penalty"] = score
    metrics["max_values"] = max_values
    return metrics


def build_daily_product_schedule_df(instance, plan_df, metrics):
    scheduled_df = plan_df[
        (plan_df["Status"] == "Scheduled")
        & (plan_df["Line"].isin(instance["final_lines"]))
    ].copy()
    scheduled_df["_line_order"] = scheduled_df["Line"].map(
        {line: index for index, line in enumerate(instance["final_lines"])}
    )

    products_by_day = {}
    max_products = 0

    for day in range(1, instance["n_days"] + 1):
        day_products = scheduled_df[scheduled_df["Day"] == day].sort_values(
            ["_line_order", "Seq."]
        )
        products_by_day[day] = day_products
        max_products = max(max_products, len(day_products))

    rows = []

    for day in range(1, instance["n_days"] + 1):
        row = {
            "Data de produção": get_production_date(instance, day),
            "Dia": day,
            "Turnos": instance.get("daily_shifts", {}).get(day, 1),
        }

        for position, (_, product) in enumerate(
            products_by_day[day].iterrows(),
            start=1,
        ):
            row[f"Linha {position}"] = product["Line"]
            row[f"Produto {position}"] = product["Reference"]
            row[f"Quantidade {position}"] = product["Master boxes"]

        for position in range(len(products_by_day[day]) + 1, max_products + 1):
            row[f"Linha {position}"] = ""
            row[f"Produto {position}"] = ""
            row[f"Quantidade {position}"] = ""

        max_utilization = 0
        max_excess = 0

        for line in instance["final_lines"]:
            production_time = metrics["production_time_by_day_line"].get(
                (day, line),
                0,
            )
            setup_time = metrics["setup_time_by_day_line"].get(
                (day, line),
                0,
            )
            available = get_available_line_time_for_day(instance, day)
            occupied = production_time + setup_time
            excess = max(0, occupied - available)
            utilization = occupied / available * 100 if available else 0
            max_excess = max(max_excess, excess)
            max_utilization = max(max_utilization, utilization)

        if max_excess > get_capacity_tolerance_for_day(instance, day):
            row["Estado"] = "Sobrecarga"
        elif max_utilization >= 90:
            row["Estado"] = "Perto do limite"
        else:
            row["Estado"] = "OK"

        rows.append(row)

    return pd.DataFrame(rows)


def build_product_matrix_df(instance, plan_df):
    scheduled_df = plan_df[
        (plan_df["Status"] == "Scheduled")
        & (plan_df["Line"].isin(instance["final_lines"]))
    ].copy()

    rows = []

    for day in range(1, instance["n_days"] + 1):
        for line in instance["final_lines"]:
            row = {
                "Production date": get_production_date(instance, day),
                "Day": day,
                "Line": line,
            }

            day_line_df = scheduled_df[
                (scheduled_df["Day"] == day)
                & (scheduled_df["Line"] == line)
            ]

            for _, item in day_line_df.iterrows():
                ref = item["Reference"]
                qty_col = f"{ref} | boxes"
                value_col = f"{ref} | euros"

                row[qty_col] = row.get(qty_col, 0) + item["Master boxes"]
                row[value_col] = row.get(value_col, 0) + item["Economic value"]

            rows.append(row)

    matrix_df = pd.DataFrame(rows).fillna(0)

    fixed_cols = ["Production date", "Day", "Line"]
    product_cols = sorted(
        col for col in matrix_df.columns
        if col not in fixed_cols
    )

    return matrix_df[fixed_cols + product_cols]


def build_time_slot_activity_df(instance, best_metrics):
    operations = best_metrics.get("time_operations", [])
    rows = []

    for day in range(1, instance["n_days"] + 1):
        standard_operators = get_standard_operators_for_day(instance, day)
        start_min = instance.get("daily_shift_start_min", {}).get(day, 8 * 60)
        end_min = instance.get("daily_shift_end_min", {}).get(day, 24 * 60)
        latest_operation_end = max(
            (
                op["end"]
                for op in operations
                if op["day"] == day
            ),
            default=end_min,
        )
        end_min = max(end_min, latest_operation_end)
        start_bucket = int(start_min // TIME_BUCKET_MIN)
        end_bucket = int((end_min + TIME_BUCKET_MIN - 1) // TIME_BUCKET_MIN)

        for bucket in range(start_bucket, end_bucket):
            start = bucket * TIME_BUCKET_MIN
            end = start + TIME_BUCKET_MIN
            active_ops = [
                op for op in operations
                if op["day"] == day and op["start"] < end and op["end"] > start
            ]

            row = {
                "Production date": get_production_date(instance, day),
                "Day": day,
                "Slot start (min)": start,
                "Time slot": f"{format_time_from_minutes(start)}-{format_time_from_minutes(end)}",
                "Standard operators": standard_operators,
            }

            total_operators = 0

            for line in instance["final_lines"]:
                line_ops = [op for op in active_ops if op["line"] == line]
                references = sorted(
                    set(
                        f"{op['ref_id']} ({op.get('master_boxes', 0)} boxes)"
                        for op in line_ops
                    )
                )
                activities = sorted(set(str(op["operation"]) for op in line_ops))
                operators = sum(op.get("operators", 0) for op in line_ops)

                row[f"{line} references"] = " | ".join(references)
                row[f"{line} activity"] = " | ".join(activities)
                row[f"{line} operators"] = operators
                total_operators += operators

            row["Total operators used"] = total_operators
            row["Operator excess"] = max(0, total_operators - standard_operators)
            row["Status"] = "Overloaded" if row["Operator excess"] > 0 else "OK"
            rows.append(row)

    return pd.DataFrame(rows)


def build_operations_df(instance, best_metrics):
    rows = []

    for op in best_metrics.get("time_operations", []):
        rows.append({
            "Production date": get_production_date(instance, op["day"]),
            "Day": op["day"],
            "Line": op["line"],
            "Reference": op["ref_id"],
            "Operation": op["operation"],
            "Master boxes": op.get("master_boxes", 0),
            "Start": format_time_from_minutes(op["start"]),
            "End": format_time_from_minutes(op["end"]),
            "Operators": op["operators"],
        })

    return pd.DataFrame(rows)


def build_penalty_df(best_metrics):
    rows = [
        {"Component": "Delay", "Penalty": best_metrics.get("delay_penalty", 0)},
        {"Component": "Postponement", "Penalty": best_metrics.get("postponement_penalty", 0)},
        {"Component": "Setup", "Penalty": best_metrics.get("setup_penalty", 0)},
        {"Component": "Economic reward", "Penalty": -best_metrics.get("economic_value_reward", 0)},
        {
            "Component": "Operator utilization reward",
            "Penalty": -best_metrics.get("operator_utilization_reward", 0),
        },
    ]

    return pd.DataFrame(rows)


def highlight_status(row):
    status = row.get("Status")
    status = row.get("Estado", status)

    if status in ["Overloaded", "Sobrecarga"]:
        return [f"background-color: {OVERLOAD_BG}; color: {OVERLOAD_TEXT}"] * len(row)

    if status in ["Near limit", "Perto do limite"]:
        return [f"background-color: {NEAR_LIMIT_BG}; color: #7a5a00"] * len(row)

    if status == "OK":
        return [f"background-color: {OK_BG}; color: {OK_TEXT}"] * len(row)

    return [f"background-color: {NEUTRAL_BG}; color: #172033"] * len(row)


def light_table_style(df):
    return (
        df.style
        .set_properties(**{
            "background-color": "#ffffff",
            "color": "#172033",
            "border-color": "#d9e0ea",
        })
        .set_table_styles([
            {
                "selector": "thead th",
                "props": [
                    ("background-color", "#153e7e"),
                    ("color", "#ffffff"),
                    ("font-weight", "700"),
                    ("border-color", "#153e7e"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(even) td",
                "props": [
                    ("background-color", "#f4f7fb"),
                    ("color", "#172033"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(odd) td",
                "props": [
                    ("background-color", "#ffffff"),
                    ("color", "#172033"),
                ],
            },
        ])
    )


def render_interactive_table(df, key, height=420):
    arrow_df = df.copy()
    arrow_df = arrow_df.replace("", pd.NA).convert_dtypes()
    display_data = light_table_style(arrow_df)

    if "Estado" in arrow_df.columns or "Status" in arrow_df.columns:
        display_data = display_data.apply(highlight_status, axis=1)

    st.dataframe(
        display_data,
        width="stretch",
        hide_index=True,
        height=height,
    )

    csv_data = arrow_df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="Descarregar tabela em CSV",
        data=csv_data,
        file_name=f"{key}.csv",
        mime="text/csv",
        key=f"download_{key}",
        width="content",
    )


st.set_page_config(
    page_title="Planeamento de Produção - Empresa X",
    layout="wide",
)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
default_excel_path = os.path.join(project_root, "Inputs_EmpresaX.xlsx")

st.markdown(
    """
    <style>
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    .main,
    .block-container {
        background-color: #eef3f8 !important;
        color: #172033 !important;
    }
    h1, h2, h3, h4, p, span, div, .stMarkdown, label {
        color: #172033;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #d9e0ea;
        border-left: 5px solid #153e7e;
        padding: 14px;
        border-radius: 8px;
        box-shadow: 0 1px 4px rgba(21, 62, 126, 0.08);
    }
    .brand-wrap {
        background-color:#ffffff;
        padding:18px 24px;
        border-radius:8px;
        margin-bottom:20px;
        border:1px solid #d9e0ea;
        border-top:6px solid #153e7e;
        box-shadow: 0 2px 8px rgba(21, 62, 126, 0.08);
    }
    .brand-line {
        display:flex;
        align-items:center;
        gap:16px;
        color:#172033;
        font-size:15px;
        font-weight:600;
        margin-bottom:10px;
    }
    .kaizen-mark {
        position:relative;
        width:86px;
        height:56px;
        background:#ffffff;
        flex:0 0 auto;
    }
    .kaizen-mark .red-triangle {
        position:absolute;
        left:10px;
        top:8px;
        width:0;
        height:0;
        border-top:18px solid #b6003b;
        border-right:36px solid transparent;
    }
    .kaizen-mark .blue-triangle {
        position:absolute;
        right:8px;
        bottom:8px;
        width:0;
        height:0;
        border-bottom:42px solid #153e7e;
        border-left:68px solid transparent;
    }
    .kaizen-word {
        color:#153e7e;
        font-size:30px;
        font-weight:900;
        letter-spacing:8px;
        line-height:0.9;
    }
    .kaizen-word small {
        display:block;
        font-size:15px;
        letter-spacing:9px;
        font-weight:700;
        margin-top:5px;
    }
    .main-title {
        color:#153e7e;
        font-size:34px;
        font-weight:800;
        letter-spacing:0;
    }
    .subtitle {
        color:#5b677a;
        font-size:18px;
        font-weight:500;
        margin-top:6px;
    }
    .stTextInput input,
    .stNumberInput input,
    textarea {
        background-color: #ffffff !important;
        color: #172033 !important;
        border: 1px solid #b8c7d9 !important;
    }
    [data-baseweb="select"],
    [data-baseweb="select"] > div,
    [data-baseweb="popover"],
    [data-baseweb="popover"] > div,
    [data-baseweb="popover"] div,
    [data-baseweb="popover"] [role="listbox"],
    [data-baseweb="popover"] [role="option"],
    [data-baseweb="menu"],
    [data-baseweb="menu"] ul,
    [data-baseweb="menu"] li,
    [data-testid="stDateInput"] input,
    [data-testid="stTimeInput"] input {
        background-color: #ffffff !important;
        color: #172033 !important;
        border-color: #b8c7d9 !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"],
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #172033 !important;
        border-color: #b8c7d9 !important;
    }
    [data-testid="stMultiSelect"] [data-baseweb="select"],
    [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #172033 !important;
        border-color: #b8c7d9 !important;
    }
    [data-baseweb="select"] *,
    [data-baseweb="popover"] *,
    [data-baseweb="menu"] *,
    [data-testid="stDateInput"] *,
    [data-testid="stTimeInput"] *,
    [data-testid="stMultiSelect"] *,
    [data-testid="stSelectbox"] *,
    [data-testid="stExpander"] *,
    [data-testid="stSlider"] * {
        color: #172033 !important;
    }
    [data-baseweb="calendar"],
    [data-baseweb="calendar"] *,
    [data-baseweb="datepicker"],
    [data-baseweb="datepicker"] *,
    [data-baseweb="calendar"] + div,
    [data-baseweb="calendar"] + div *,
    [data-baseweb="calendar"] footer,
    [data-baseweb="calendar"] footer *,
    [role="dialog"],
    [role="dialog"] * {
        background-color: #ffffff !important;
        color: #172033 !important;
        border-color: #b8c7d9 !important;
    }
    [data-baseweb="calendar"] input,
    [data-baseweb="datepicker"] input {
        background-color: #ffffff !important;
        color: #172033 !important;
        border: 1px solid #b8c7d9 !important;
    }
    [data-baseweb="calendar"] table,
    [data-baseweb="calendar"] tbody,
    [data-baseweb="calendar"] tr,
    [data-baseweb="calendar"] td,
    [data-baseweb="calendar"] button,
    [data-baseweb="calendar"] [role="grid"],
    [data-baseweb="calendar"] [role="row"],
    [data-baseweb="calendar"] [role="gridcell"],
    [data-baseweb="calendar"] [aria-disabled="true"],
    [data-baseweb="calendar"] button:disabled,
    [data-baseweb="calendar"] [disabled] {
        background: #ffffff !important;
        background-color: #ffffff !important;
        color: #172033 !important;
    }
    [data-baseweb="calendar"] [aria-disabled="true"] *,
    [data-baseweb="calendar"] button:disabled *,
    [data-baseweb="calendar"] [disabled] * {
        background: #ffffff !important;
        background-color: #ffffff !important;
        color: #ffffff !important;
    }
    [data-baseweb="calendar"] div:empty,
    [data-baseweb="calendar"] span:empty,
    [data-baseweb="datepicker"] div:empty,
    [data-baseweb="datepicker"] span:empty,
    div[data-baseweb="popover"] div:empty,
    div[data-baseweb="popover"] span:empty {
        background: #ffffff !important;
        background-color: #ffffff !important;
        border-color: #ffffff !important;
        box-shadow: none !important;
        outline: none !important;
    }
    [data-baseweb="calendar"] *::before,
    [data-baseweb="calendar"] *::after,
    [data-baseweb="datepicker"] *::before,
    [data-baseweb="datepicker"] *::after {
        background: transparent !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }
    [aria-selected="true"],
    [data-baseweb="calendar"] button[aria-selected="true"] {
        background-color: #153e7e !important;
        color: #ffffff !important;
    }
    [data-baseweb="calendar"] button:hover,
    [data-baseweb="menu"] li:hover,
    [data-baseweb="popover"] [role="option"]:hover {
        background-color: #eef3f8 !important;
        color: #153e7e !important;
    }
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] [role="listbox"],
    div[data-baseweb="popover"] [role="listbox"] > div,
    div[data-baseweb="popover"] [role="option"],
    div[data-baseweb="popover"] [data-baseweb="menu"],
    div[data-baseweb="popover"] [data-baseweb="menu"] ul,
    div[data-baseweb="popover"] [data-baseweb="menu"] li {
        background: #ffffff !important;
        background-color: #ffffff !important;
        color: #172033 !important;
        border-color: #b8c7d9 !important;
    }
    div[data-baseweb="popover"] [role="listbox"] *,
    div[data-baseweb="popover"] [role="option"] *,
    div[data-baseweb="popover"] [data-baseweb="menu"] * {
        color: #172033 !important;
    }
    div[data-baseweb="popover"] [role="option"]:hover,
    div[data-baseweb="popover"] [data-baseweb="menu"] li:hover {
        background: #eef3f8 !important;
        background-color: #eef3f8 !important;
        color: #153e7e !important;
    }
    div[data-baseweb="popover"] [aria-selected="true"],
    div[data-baseweb="popover"] [aria-selected="true"] * {
        background: #153e7e !important;
        background-color: #153e7e !important;
        color: #ffffff !important;
    }
    [data-baseweb="tag"] {
        background-color: #eef3f8 !important;
        color: #153e7e !important;
        border: 1px solid #b8c7d9 !important;
    }
    [data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #cfd9e6 !important;
        border-radius: 8px !important;
    }
    [data-testid="stExpander"] details,
    [data-testid="stExpander"] summary {
        background-color: #ffffff !important;
    }
    [data-testid="stSlider"] [role="slider"] {
        background-color: #153e7e !important;
        border-color: #153e7e !important;
    }
    [data-testid="stSlider"] div[data-baseweb="slider"] {
        background: transparent !important;
    }
    [data-testid="stSlider"] div[data-baseweb="slider"] [aria-hidden="true"] {
        background-color: #d7e1ef !important;
    }
    .stAlert,
    [data-testid="stAlert"] {
        background-color: #ffffff !important;
        color: #172033 !important;
        border: 1px solid #d9e0ea !important;
    }
    [data-testid="stAlert"] * {
        color: #172033 !important;
    }
    .stFileUploader,
    [data-testid="stFileUploader"] {
        background-color: #ffffff !important;
        color: #172033 !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploader"] * {
        color: #172033 !important;
    }
    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploader"] button * {
        color: #ffffff !important;
    }
    [data-testid="stDataFrame"],
    [data-testid="stDataFrame"] > div,
    [data-testid="stDataEditor"],
    [data-testid="stDataEditor"] > div,
    [data-testid="stDataEditor"] div[role="grid"],
    [data-testid="stDataEditor"] div[role="gridcell"],
    [data-testid="stDataEditor"] div[role="columnheader"] {
        background-color: #ffffff !important;
        color: #172033 !important;
        border-color: #cfd9e6 !important;
    }
    [data-testid="stDataFrame"] button,
    [data-testid="stDataFrame"] button *,
    [data-testid="stDataEditor"] button,
    [data-testid="stDataEditor"] button * {
        color: #153e7e !important;
    }
    [data-testid="stDataEditor"] input,
    [data-testid="stDataEditor"] textarea {
        background-color: #ffffff !important;
        color: #172033 !important;
        caret-color: #153e7e !important;
    }
    [data-testid="stDataEditor"] canvas {
        background-color: #ffffff !important;
    }
    .kaizen-table-wrap {
        width: 100%;
        overflow: auto;
        background: #ffffff;
        border: 1px solid #cfd9e6;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(21, 62, 126, 0.08);
        margin-bottom: 24px;
    }
    table.kaizen-table {
        border-collapse: collapse;
        width: max-content;
        min-width: 100%;
        background: #ffffff;
        color: #172033;
        font-size: 13px;
    }
    table.kaizen-table th {
        position: sticky;
        top: 0;
        z-index: 2;
        background: #153e7e;
        color: #ffffff !important;
        padding: 10px 12px;
        border: 1px solid #153e7e;
        text-align: left;
        white-space: nowrap;
        font-weight: 800;
    }
    table.kaizen-table td {
        background: #ffffff;
        color: #172033 !important;
        padding: 9px 12px;
        border: 1px solid #d9e0ea;
        vertical-align: top;
        white-space: nowrap;
    }
    table.kaizen-table tr:nth-child(even) td {
        background: #f4f7fb;
    }
    .stButton button,
    button[kind="primary"],
    button[kind="secondary"],
    button[data-testid="stBaseButton-primary"],
    button[data-testid="stBaseButton-secondary"],
    div[data-testid="stButton"] button,
    div[data-testid="stFileUploader"] button {
        background: #153e7e !important;
        background-color: #153e7e !important;
        color: #ffffff !important;
        border: 1px solid #153e7e !important;
        box-shadow: none !important;
    }
    .stButton button *,
    button[kind="primary"] *,
    button[kind="secondary"] *,
    button[data-testid="stBaseButton-primary"] *,
    button[data-testid="stBaseButton-secondary"] *,
    div[data-testid="stButton"] button *,
    div[data-testid="stFileUploader"] button * {
        color: #ffffff !important;
    }
    .stButton button:hover,
    button[kind="primary"]:hover,
    button[kind="secondary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover,
    button[data-testid="stBaseButton-secondary"]:hover,
    div[data-testid="stButton"] button:hover,
    div[data-testid="stFileUploader"] button:hover {
        background: #b6003b !important;
        background-color: #b6003b !important;
        color: #ffffff !important;
        border-color: #b6003b !important;
    }
    /* Light Kaizen readability fixes */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main {
        background-color: #eef3f8 !important;
        color: #172033 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #ccd7e6 !important;
        border-left: 6px solid #153e7e !important;
        box-shadow: 0 2px 8px rgba(21, 62, 126, 0.10) !important;
    }
    div[data-testid="stMetric"] * {
        color: #153e7e !important;
    }
    div[data-testid="stMetricLabel"] * {
        color: #5b677a !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricValue"] * {
        color: #153e7e !important;
        font-weight: 900 !important;
    }
    .st-emotion-cache-1r6slb0,
    .st-emotion-cache-1wmy9hl,
    .st-emotion-cache-13k62yr {
        background-color: #eef3f8 !important;
    }
    </style>
    <div class="brand-wrap">
        <div class="brand-line">
            <span class="kaizen-mark"><span class="blue-triangle"></span><span class="red-triangle"></span></span>
            <span class="kaizen-word">KAIZEN<small>INSTITUTE</small></span>
        </div>        <div class="main-title">PLANEAMENTO DE PRODUÇÃO - DOCELEIA</div>
        <div class="subtitle">Planeamento mensal de produção</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Ficheiro de inputs")
input_col1, input_col2 = st.columns([2, 1])

with input_col1:
    excel_path = st.text_input(
        "Caminho do ficheiro Excel",
        value=default_excel_path,
    )

with input_col2:
    uploaded_excel = st.file_uploader(
        "Ou carregar ficheiro Excel",
        type=["xlsx"],
    )

excel_source = uploaded_excel if uploaded_excel is not None else excel_path

try:
    base_instance = load_real_instance(excel_source)
except Exception as exc:
    st.error(f"Não foi possível carregar o ficheiro de inputs: {exc}")
    st.stop()

default_working_days = base_instance.get("working_days", [])

if default_working_days:
    default_start_date = default_working_days[0]
    default_end_date = default_working_days[-1]
else:
    default_start_date = date.today()
    default_end_date = date.today()

st.subheader("Parâmetros operacionais do plano")

horizon_col1, horizon_col2 = st.columns(2)

with horizon_col1:
    planning_start_date = st.date_input(
        "Início do horizonte de planeamento",
        value=default_start_date,
    )

with horizon_col2:
    planning_end_date = st.date_input(
        "Fim do horizonte de planeamento",
        value=default_end_date,
    )

if planning_end_date < planning_start_date:
    st.error("A data final não pode ser anterior à data inicial.")
    st.stop()

all_calendar_days = []
current_day = planning_start_date

while current_day <= planning_end_date:
    all_calendar_days.append(current_day)
    current_day += timedelta(days=1)

non_working_days = st.multiselect(
    "Dias de limpeza / feriados",
    options=all_calendar_days,
    format_func=format_date,
    default=[],
    placeholder="Selecionar dias",
)

working_days = build_working_days_from_dashboard(
    planning_start_date,
    planning_end_date,
    non_working_days,
)

if not working_days:
    st.error("O horizonte selecionado não tem dias úteis para planear.")
    st.stop()

default_operators = int(base_instance.get("standard_operators", 0) or 0)
operators_by_day = {}
start_times_by_day = {}
end_times_by_day = {}
shifts_by_day = {}

with st.expander("Operadores e horários por dia", expanded=True):
    st.caption(
        "Os valores abaixo usam os defaults como ponto de partida, "
        "mas podem ser ajustados para cada dia do plano."
    )

    for day_index, working_day in enumerate(working_days, start=1):
        day_col1, day_col2, day_col3, day_col4, day_col5 = st.columns([1.4, 1, 1, 1, 1])

        with day_col1:
            st.markdown(f"**{day_index} - {format_date(working_day)}**")

        with day_col2:
            operators_by_day[day_index] = st.slider(
                "Operadores",
                min_value=0,
                max_value=max(40, default_operators + 20),
                value=default_operators,
                step=1,
                key=f"operators_day_{day_index}",
            )

        with day_col3:
            shifts_by_day[day_index] = st.selectbox(
                "Turnos",
                options=[1, 2],
                index=0,
                key=f"shifts_day_{day_index}",
            )

        with day_col4:
            start_times_by_day[day_index] = st.time_input(
                "Início",
                value=time(8, 0),
                key=f"start_time_day_{day_index}",
            )

        with day_col5:
            end_times_by_day[day_index] = st.time_input(
                "Fim",
                value=time(16, 30),
                key=f"end_time_day_{day_index}",
            )

instance = apply_dashboard_overrides(
    base_instance,
    working_days,
    operators_by_day,
    start_times_by_day,
    end_times_by_day,
    shifts_by_day,
)
current_instance_signature = build_instance_signature(instance)

scenario_rows = []

for day_index, working_day in enumerate(working_days, start=1):
    scenario_rows.append({
        "Dia": day_index,
        "Data": format_date(working_day),
        "Operadores disponíveis": operators_by_day[day_index],
        "Turnos": shifts_by_day[day_index],
        "Início": start_times_by_day[day_index].strftime("%H:%M"),
        "Fim": end_times_by_day[day_index].strftime("%H:%M"),
        "Capacidade disponível (min)": instance["daily_capacity_min"][day_index],
    })

st.subheader("Resumo dos parâmetros do cenário")
st.dataframe(
    pd.DataFrame(scenario_rows),
    width="stretch",
    hide_index=True,
)

run_button = st.button("Gerar plano de produção", width="content")

if run_button:
    try:
        with st.spinner("A carregar dados e a correr o algoritmo genético..."):
            planning_month = get_planning_month(instance)
            best_solution, best_metrics, actual_generations = run_genetic_algorithm(
                instance,
                population_size=POPULATION_SIZE,
                generations=GENERATIONS,
                mutation_rate=MUTATION_RATE,
                elite_size=ELITE_SIZE,
                tournament_size=TOURNAMENT_SIZE,
                seed=RANDOM_SEED,
            )

        st.session_state["ga_solution"] = deepcopy(best_solution)
        st.session_state["ga_metrics"] = deepcopy(best_metrics)
        st.session_state["ga_planning_month"] = planning_month
        st.session_state["ga_instance_signature"] = current_instance_signature
        st.session_state["scenario_version"] = (
            st.session_state.get("scenario_version", 0) + 1
        )
    except Exception as exc:
        st.error(f"Não foi possível gerar o plano: {exc}")
        st.stop()

if "ga_solution" not in st.session_state:
    st.info("Pressione o botão para gerar o plano de produção.")
    st.stop()

if st.session_state.get("ga_instance_signature") != current_instance_signature:
    for session_key in [
        "ga_solution",
        "ga_metrics",
        "ga_planning_month",
        "ga_instance_signature",
    ]:
        st.session_state.pop(session_key, None)

    st.warning(
        "Os inputs ou os parâmetros operacionais foram alterados. "
        "Gere novamente o plano para evitar misturar uma solução antiga "
        "com a instância atual."
    )
    st.stop()

best_solution = deepcopy(st.session_state["ga_solution"])
duplicate_orders, missing_orders, unexpected_orders = validate_solution_orders(
    best_solution,
    instance,
)

if duplicate_orders or missing_orders or unexpected_orders:
    st.error(
        "A solução guardada não contém exatamente uma ocorrência de cada ordem. "
        f"Duplicadas: {duplicate_orders or 'nenhuma'}; "
        f"em falta: {missing_orders or 'nenhuma'}; "
        f"inesperadas: {unexpected_orders or 'nenhuma'}. "
        "Gere novamente o plano."
    )
    st.stop()

max_values = compute_max_values(instance)
best_metrics = add_normalised_fitness_metrics(
    evaluate_solution(best_solution, instance),
    max_values,
)
planning_month = st.session_state.get(
    "ga_planning_month",
    get_planning_month(instance),
)

st.success(f"Plano gerado com sucesso para {planning_month}.")

plan_df = build_plan_df(instance, best_solution)
daily_product_schedule_df = build_daily_product_schedule_df(
    instance,
    plan_df,
    best_metrics,
)
product_matrix_df = build_product_matrix_df(instance, plan_df)
capacity_df = build_capacity_df(instance, best_metrics)
time_slot_df = build_time_slot_activity_df(instance, best_metrics)
operations_df = build_operations_df(instance, best_metrics)

st.subheader("Indicadores principais")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Kg produzidos", f"{best_metrics.get('scheduled_kg', 0):,.1f}")
col2.metric("Kg adiados", f"{best_metrics.get('postponed_kg', 0):,.1f}")
col3.metric("Valor produzido", f"€{best_metrics.get('scheduled_economic_value', 0):,.0f}")
col4.metric("Valor adiado", f"€{best_metrics.get('postponed_economic_value', 0):,.0f}")

st.subheader("Teste interativo de cenários")
st.caption(
    "Altere o dia, a sequência ou o estado de adiamento de uma ordem. "
    "A linha permanece bloqueada por ser uma restrição dura. "
    "As métricas e a penalização são recalculadas automaticamente."
)

reset_scenario = st.button(
    "Repor solução do GA",
    width="content",
)

if reset_scenario:
    st.session_state["scenario_version"] = (
        st.session_state.get("scenario_version", 0) + 1
    )
    st.rerun()

scenario_df = build_scenario_editor_df(instance, best_solution)
scenario_version = st.session_state.get("scenario_version", 0)
edited_scenario_df = st.data_editor(
    scenario_df,
    column_config={
        "ID da ordem": st.column_config.NumberColumn(format="%d"),
        "Caixas master": st.column_config.NumberColumn(format="%d"),
        "Dia": st.column_config.NumberColumn(
            min_value=1,
            max_value=instance["n_days"],
            step=1,
            format="%d",
        ),
        "Sequência": st.column_config.NumberColumn(
            min_value=1,
            step=1,
            format="%d",
        ),
        "Adiado": st.column_config.CheckboxColumn(),
        "Valor económico": st.column_config.NumberColumn(
            format="€ %.2f",
        ),
    },
    disabled=[
        "ID da ordem",
        "Referência",
        "Caixas master",
        "Linha",
        "Dia de entrega",
        "Valor económico",
    ],
    hide_index=True,
    width="stretch",
    height=430,
    key=f"scenario_editor_{scenario_version}",
)

manual_solution, scenario_errors = build_solution_from_scenario(
    instance,
    best_solution,
    edited_scenario_df,
)

if scenario_errors:
    st.error(
        "O cenário manual tem alterações inválidas:\n\n- "
        + "\n- ".join(scenario_errors)
    )
else:
    manual_metrics = add_normalised_fitness_metrics(
        evaluate_solution(manual_solution, instance),
        max_values,
    )
    comparison_df = build_scenario_comparison_df(
        best_metrics,
        manual_metrics,
    )

    comparison_col1, comparison_col2 = st.columns(2)
    comparison_col1.metric(
        "Fitness normalizada do GA",
        f"{best_metrics.get('total_penalty', 0):,.6f}",
    )
    comparison_col2.metric(
        "Fitness normalizada do cenário manual",
        f"{manual_metrics.get('total_penalty', 0):,.6f}",
        delta=(
            f"{manual_metrics.get('total_penalty', 0) - best_metrics.get('total_penalty', 0):,.6f}"
        ),
        delta_color="inverse",
    )

    render_interactive_table(
        comparison_df,
        key="comparacao_ga_cenario_manual",
        height=330,
    )

    manual_plan_df = build_plan_df(instance, manual_solution)
    manual_schedule_display = build_daily_product_schedule_df(
        instance,
        manual_plan_df,
        manual_metrics,
    )
    manual_capacity_df = build_capacity_df(instance, manual_metrics)

    st.markdown("**Sequência resultante do cenário manual**")
    render_interactive_table(
        manual_schedule_display,
        key="sequencia_cenario_manual",
        height=360,
    )

    st.markdown("**Capacidade resultante do cenário manual**")
    manual_capacity_display = manual_capacity_df.rename(columns={
        "Production date": "Data de produção",
        "Day": "Dia",
        "Line": "Linha",
        "Shifts": "Turnos",
        "Production time (min)": "Tempo de produção (min)",
        "Setup time (min)": "Tempo de setup (min)",
        "Occupied time (min)": "Tempo ocupado (min)",
        "Available time (min)": "Tempo disponível (min)",
        "Capacity excess (min)": "Excesso de capacidade (min)",
        "Utilization (%)": "Utilização (%)",
    })
    render_interactive_table(
        manual_capacity_display,
        key="capacidade_cenario_manual",
        height=360,
    )

st.subheader("Sequência diária de produção")
render_interactive_table(
    daily_product_schedule_df,
    key="sequencia_diaria_producao",
    height=420,
)

st.subheader("Quantidade e valor por produto, dia e linha")
product_matrix_display_df = product_matrix_df.rename(columns={
    "Production date": "Data de produção",
    "Day": "Dia",
    "Line": "Linha",
})
render_interactive_table(
    product_matrix_display_df,
    key="quantidade_valor_produto_dia_linha",
    height=520,
)

st.subheader("Ocupação de operadores por horário")
if time_slot_df.empty:
    st.info("Não existe simulação horária disponível.")
else:
    hourly_chart = (
        alt.Chart(time_slot_df)
        .mark_rect()
        .encode(
            x=alt.X(
                "Time slot:N",
                title="Horário",
                sort=alt.SortField(
                    field="Slot start (min)",
                    order="ascending",
                ),
            ),
            y=alt.Y("Production date:N", title="Data de produção"),
            color=alt.Color(
                "Total operators used:Q",
                title="Operadores usados",
                scale=alt.Scale(
                    domain=[0, max(1, time_slot_df["Total operators used"].max())],
                    range=["#f1f5f9", "#153e7e", "#b6003b"],
                ),
            ),
            tooltip=list(time_slot_df.columns),
        )
        .properties(height=max(320, 24 * time_slot_df["Production date"].nunique()))
    )
    st.altair_chart(hourly_chart, width="stretch")

    operators_df = time_slot_df.rename(columns={
        "Production date": "Data de produção",
        "Day": "Dia",
        "Time slot": "Horário",
        "Standard operators": "Operadores disponíveis",
        "L1 references": "Referências L1",
        "L1 activity": "Atividade L1",
        "L1 operators": "Operadores L1",
        "L2 references": "Referências L2",
        "L2 activity": "Atividade L2",
        "L2 operators": "Operadores L2",
        "Total operators used": "Operadores usados",
        "Operator excess": "Excesso de operadores",
        "Status": "Estado",
    })
    operators_df = operators_df.drop(
        columns=["Slot start (min)"],
        errors="ignore",
    )
    for col in ["Atividade L1", "Atividade L2"]:
        if col in operators_df.columns:
            operators_df[col] = operators_df[col].replace({
                "production": "produção",
                "finishing": "acabamento",
                "setup": "setup",
                "production | setup": "produção | setup",
                "finishing | production": "acabamento | produção",
                "finishing | setup": "acabamento | setup",
                "finishing | production | setup": "acabamento | produção | setup",
            }, regex=False)
    operators_df["Estado"] = operators_df["Estado"].replace({
        "Overloaded": "Sobrecarga",
        "OK": "OK",
    })
    render_interactive_table(
        operators_df,
        key="ocupacao_operadores_horario",
        height=650,
    )

st.subheader("Utilização de capacidade por linha")
if capacity_df.empty:
    st.info("Não existe utilização de capacidade registada.")
else:
    capacity_chart = (
        alt.Chart(capacity_df)
        .mark_bar()
        .encode(
            x=alt.X("Production date:N", title="Data de produção"),
            xOffset=alt.XOffset("Line:N"),
            y=alt.Y("Utilization (%):Q", title="Utilização (%)"),
            color=alt.Color("Line:N", title="Linha", scale=alt.Scale(range=LINE_COLORS)),
            tooltip=list(capacity_df.columns),
        )
        .properties(height=320)
    )
    st.altair_chart(capacity_chart, width="stretch")

    capacity_display_df = capacity_df.rename(columns={
        "Production date": "Data de produção",
        "Day": "Dia",
        "Line": "Linha",
        "Shifts": "Turnos",
        "Production time (min)": "Tempo de produção (min)",
        "Setup time (min)": "Tempo de setup (min)",
        "Occupied time (min)": "Tempo ocupado (min)",
        "Available time (min)": "Tempo disponível (min)",
        "Capacity excess (min)": "Excesso de capacidade (min)",
        "Utilization (%)": "Utilização (%)",
    })
    render_interactive_table(
        capacity_display_df,
        key="utilizacao_capacidade_linha",
        height=420,
    )

st.subheader("Plano detalhado")
plan_display_df = plan_df.rename(columns={
    "Status": "Estado",
    "Production date": "Data de produção",
    "Day": "Dia",
    "Line": "Linha",
    "Seq.": "Seq.",
    "Reference": "Referência",
    "Family": "Família",
    "Master boxes": "Caixas master",
    "Kg": "Kg",
    "Economic value": "Valor económico",
    "Production time (min)": "Tempo de produção (min)",
    "Setup time (min)": "Tempo de setup (min)",
    "Delivery date": "Data de entrega",
    "Delivery day": "Dia de entrega",
})
plan_display_df["Estado"] = plan_display_df["Estado"].replace({
    "Scheduled": "Planeado",
    "Postponed": "Adiado",
})
render_interactive_table(
    plan_display_df,
    key="plano_detalhado",
    height=520,
)

st.subheader("Operações por horário")
if operations_df.empty:
    st.info("Não existe timeline de operações disponível.")
else:
    operations_display_df = operations_df.rename(columns={
        "Production date": "Data de produção",
        "Day": "Dia",
        "Line": "Linha",
        "Reference": "Referência",
        "Operation": "Operação",
        "Master boxes": "Caixas master",
        "Start": "Início",
        "End": "Fim",
        "Operators": "Operadores",
    })
    operations_display_df["Operação"] = operations_display_df["Operação"].replace({
        "production": "produção",
        "finishing": "acabamento",
        "setup": "setup",
    })
    render_interactive_table(
        operations_display_df,
        key="operacoes_horario",
        height=520,
    )
