import os

import altair as alt
import pandas as pd
import streamlit as st

from generate_instance import load_real_instance
from geneticalgorithm import run_genetic_algorithm
from evaluator import (
    create_refs_by_id,
    get_production_time,
    get_setup,
    TIME_BUCKET_MIN,
)


LINE_COLORS = ["#38bdf8", "#22c55e"]
OVERLOAD_BG = "#4c1d1d"
OVERLOAD_TEXT = "#fecaca"
NEUTRAL_BG = "#111827"
NEAR_LIMIT_BG = "#1f2937"

POPULATION_SIZE = 100
GENERATIONS = 100
MUTATION_RATE = 0.10
ELITE_SIZE = 5
TOURNAMENT_SIZE = 3
RANDOM_SEED = 42


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
    day_offset = value // (24 * 60)
    value = value % (24 * 60)
    hours = value // 60
    minutes = value % 60

    if day_offset > 0:
        return f"+{day_offset}d {hours:02d}:{minutes:02d}"

    return f"{hours:02d}:{minutes:02d}"


def get_production_date(instance, day):
    if day is None:
        return "Postponed"

    working_days = instance.get("working_days", [])

    if working_days and 1 <= day <= len(working_days):
        return format_date(working_days[day - 1])

    return f"Day {day}"


def get_planning_month(instance):
    working_days = instance.get("working_days", [])

    if working_days:
        return working_days[0].strftime("%B %Y")

    return "Planning horizon"


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
            "Production time (min)": round(production_time, 1),
            "Setup time (min)": round(setup_time, 1),
            "Delivery date": get_delivery_date(instance, item),
            "Delivery day": item.get("delivery_date", ""),
        })

    return pd.DataFrame(plan_rows)


def build_capacity_df(instance, best_metrics):
    rows = []
    keys = set(best_metrics["production_time_by_day_line"].keys())
    keys.update(best_metrics["setup_time_by_day_line"].keys())

    for day, line in sorted(keys):
        production_time = best_metrics["production_time_by_day_line"].get((day, line), 0)
        setup_time = best_metrics["setup_time_by_day_line"].get((day, line), 0)
        occupied_time = production_time + setup_time
        available_time = instance["available_line_time_min"]
        excess = best_metrics["capacity_excess_by_day_line"].get((day, line), 0)
        utilization = occupied_time / available_time * 100 if available_time else 0

        rows.append({
            "Production date": get_production_date(instance, day),
            "Day": day,
            "Line": line,
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
        }

        max_utilization = 0
        total_excess = 0

        for line in instance["final_lines"]:
            line_df = day_df[day_df["Line"] == line].sort_values("Seq.")
            refs = " -> ".join(line_df["Reference"].astype(str).tolist())
            production_time = best_metrics["production_time_by_day_line"].get((day, line), 0)
            setup_time = best_metrics["setup_time_by_day_line"].get((day, line), 0)
            excess = best_metrics["capacity_excess_by_day_line"].get((day, line), 0)
            available = instance["available_line_time_min"]
            utilization = (production_time + setup_time) / available * 100 if available else 0

            row[f"{line} sequence"] = refs
            row[f"{line} production"] = f"{production_time:.1f} min"
            row[f"{line} setup"] = f"{setup_time:.1f} min"
            row[f"{line} excess"] = f"{excess:.1f} min"

            max_utilization = max(max_utilization, utilization)
            total_excess += excess

        if total_excess > 0:
            row["Status"] = "Overloaded"
        elif max_utilization >= 90:
            row["Status"] = "Near limit"
        else:
            row["Status"] = "OK"

        rows.append(row)

    return pd.DataFrame(rows)


def build_time_slot_activity_df(instance, best_metrics):
    operations = best_metrics.get("time_operations", [])
    standard_operators = best_metrics.get("standard_operators", instance.get("standard_operators", 0))
    rows = []
    start_bucket = (8 * 60) // TIME_BUCKET_MIN
    end_bucket = (24 * 60) // TIME_BUCKET_MIN

    for day in range(1, instance["n_days"] + 1):
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
                "Time slot": f"{format_time_from_minutes(start)}-{format_time_from_minutes(end)}",
                "Standard operators": standard_operators,
            }

            total_operators = 0

            for line in instance["final_lines"]:
                line_ops = [op for op in active_ops if op["line"] == line]
                references = sorted(set(str(op["ref_id"]) for op in line_ops))
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
            "Start": format_time_from_minutes(op["start"]),
            "End": format_time_from_minutes(op["end"]),
            "Operators": op["operators"],
        })

    return pd.DataFrame(rows)


def build_penalty_df(best_metrics):
    rows = [
        {"Component": "Capacity", "Penalty": best_metrics.get("total_capacity_excess", 0)},
        {"Component": "Delay", "Penalty": best_metrics.get("delay_days_total", 0)},
        {"Component": "Postponement", "Penalty": best_metrics.get("postponement_penalty", 0)},
        {"Component": "Hourly operators", "Penalty": best_metrics.get("hourly_operator_penalty", 0)},
        {"Component": "Setup", "Penalty": best_metrics.get("setup_penalty", 0)},
    ]

    return pd.DataFrame(rows)


def highlight_status(row):
    status = row.get("Status")

    if status == "Overloaded":
        return [f"background-color: {OVERLOAD_BG}; color: {OVERLOAD_TEXT}"] * len(row)

    if status == "Near limit":
        return [f"background-color: {NEAR_LIMIT_BG}; color: #facc15"] * len(row)

    return [f"background-color: {NEUTRAL_BG}; color: #d1d5db"] * len(row)


st.set_page_config(
    page_title="Production Sequencing Dashboard",
    layout="wide",
)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
default_excel_path = os.path.join(project_root, "Inputs_Doceleia.xlsx")

try:
    instance = load_real_instance(default_excel_path)
except Exception as exc:
    st.error(f"Could not load Excel instance: {exc}")
    st.stop()

planning_month = get_planning_month(instance)

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #0b0f17; color: #e5e7eb; }}
    div[data-testid="stMetric"] {{
        background-color: #111827;
        border: 1px solid #334155;
        padding: 14px;
        border-radius: 8px;
    }}
    </style>
    <div style="background-color:#111827;padding:18px 24px;border-radius:8px;margin-bottom:20px;border:1px solid #334155;">
        <div style="color:#9ca3af;font-size:14px;font-weight:500;margin-bottom:4px;">Kaizen Institute</div>
        <div style="color:#f9fafb;font-size:34px;font-weight:800;letter-spacing:0;">PRODUCTION SCHEDULING - X COMPANY</div>
        <div style="color:#cbd5e1;font-size:18px;font-weight:500;margin-top:6px;">Planning month: {planning_month}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

run_button = st.button("Run Genetic Algorithm", width="content")

if run_button:
    with st.spinner("Running genetic algorithm..."):
        best_solution, best_metrics = run_genetic_algorithm(
            instance,
            population_size=POPULATION_SIZE,
            generations=GENERATIONS,
            mutation_rate=MUTATION_RATE,
            elite_size=ELITE_SIZE,
            tournament_size=TOURNAMENT_SIZE,
            seed=RANDOM_SEED,
        )

    st.success("Genetic algorithm finished.")

    plan_df = build_plan_df(instance, best_solution)
    compact_df = build_compact_schedule_df(instance, plan_df, best_metrics)
    capacity_df = build_capacity_df(instance, best_metrics)
    time_slot_df = build_time_slot_activity_df(instance, best_metrics)
    operations_df = build_operations_df(instance, best_metrics)
    penalty_df = build_penalty_df(best_metrics)

    st.subheader("Solution Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total penalty", f"{best_metrics['total_penalty']:.0f}")
    col2.metric("Postponed orders", f"{best_metrics.get('postponed_orders', 0)}")
    col3.metric("Capacity excess", f"{best_metrics['total_capacity_excess']:.1f} min")
    col4.metric("Hourly operator excess", f"{best_metrics.get('total_hourly_operator_excess', 0):.0f}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Delay", f"{best_metrics['delay_days_total']} days")
    col6.metric("Daily operator excess", f"{best_metrics['total_operator_excess']:.0f}")
    col7.metric("Peak operators", f"{best_metrics.get('peak_operators', 0):.0f}")
    col8.metric("Setup time", f"{best_metrics['setup_total_min']:.0f} min")

    st.subheader("Penalty Breakdown")
    penalty_chart = (
        alt.Chart(penalty_df)
        .mark_bar(color="#64748b")
        .encode(
            x=alt.X("Penalty:Q", title="Penalty value"),
            y=alt.Y("Component:N", sort="-x", title=""),
            tooltip=["Component", alt.Tooltip("Penalty:Q", format=".1f")],
        )
        .properties(height=220)
    )
    st.altair_chart(penalty_chart, width="stretch")

    st.subheader("Daily Production Schedule")
    st.dataframe(
        compact_df.style.apply(highlight_status, axis=1),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Capacity Usage by Line")
    if capacity_df.empty:
        st.info("No capacity usage registered.")
    else:
        capacity_chart = (
            alt.Chart(capacity_df)
            .mark_bar()
            .encode(
                x=alt.X("Production date:N", title="Production date"),
                xOffset=alt.XOffset("Line:N"),
                y=alt.Y("Utilization (%):Q", title="Utilization (%)"),
                color=alt.Color("Line:N", scale=alt.Scale(range=LINE_COLORS)),
                tooltip=list(capacity_df.columns),
            )
            .properties(height=350)
        )
        st.altair_chart(capacity_chart, width="stretch")
        st.dataframe(capacity_df, width="stretch", hide_index=True)

    st.subheader("Full Hourly Production and Operator View")
    if time_slot_df.empty:
        st.info("No hourly simulation available.")
    else:
        hourly_heatmap = (
            alt.Chart(time_slot_df)
            .mark_rect()
            .encode(
                x=alt.X("Time slot:N", title="Time slot"),
                y=alt.Y("Production date:N", title="Production date"),
                color=alt.Color(
                    "Operator excess:Q",
                    title="Operator excess",
                    scale=alt.Scale(domain=[0, max(1, time_slot_df["Operator excess"].max())], range=["#111827", "#ef4444"]),
                ),
                tooltip=list(time_slot_df.columns),
            )
            .properties(height=max(320, 24 * time_slot_df["Production date"].nunique()))
        )
        st.altair_chart(hourly_heatmap, width="stretch")
        st.markdown("Complete hourly table")
        st.dataframe(
            time_slot_df.style.apply(highlight_status, axis=1),
            width="stretch",
            hide_index=True,
            height=650,
        )

    st.subheader("Detailed Production Plan")
    st.dataframe(plan_df, width="stretch", hide_index=True)

    st.subheader("Operation Timeline Data")
    if operations_df.empty:
        st.info("No operation-level timeline available.")
    else:
        st.dataframe(operations_df, width="stretch", hide_index=True)
else:
    st.info("Press Run Genetic Algorithm to generate the product    ion schedule.")
