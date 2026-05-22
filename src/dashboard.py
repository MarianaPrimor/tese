import os
import streamlit as st
import pandas as pd
import altair as alt

from generate_instance import load_real_instance
from geneticalgorithm import run_genetic_algorithm
from evaluator import (
    create_refs_by_id,
    get_production_time,
    get_required_operators,
    get_setup,
    TIME_BUCKET_MIN,
    INVALID_LINE_PENALTY,
    CAPACITY_PENALTY,
    DELAY_PENALTY,
)


LINE_COLORS = ["#ef4444", "#22c55e"]
STATUS_COLORS = {
    "Overloaded": "#4c1d1d",
    "Near limit": "#3f2f12",
    "OK": "#123524",
}


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
    hours = value // 60
    minutes = value % 60
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

    for original_index, s in sorted_solution:
        if s.get("postponed"):
            continue

        day = s.get("day")
        line = s.get("line") or "INVALID"
        ref_id = str(s.get("ref_id")).strip()
        key = (day, line)

        sequence_by_day_line[key] = sequence_by_day_line.get(key, 0) + 1
        ref = refs_by_id.get(ref_id)

        if ref is not None and line in ["L1", "L2"]:
            production_time = get_production_time(
                ref,
                line,
                s.get("master_boxes"),
            ) or 0
            operators = get_required_operators(ref, line)
            setup_time = get_setup(
                instance,
                last_family_by_day_line.get(key),
                ref["family"],
            )
            last_family_by_day_line[key] = ref["family"]
            family = ref["family"]
        else:
            production_time = 0
            operators = 0
            setup_time = 0
            family = ""

        delivery_day = s.get("delivery_date", day)
        delay = max(0, day - delivery_day) if day is not None else ""

        plan_rows.append({
            "Production date": get_production_date(instance, day),
            "Day": day,
            "Line": line,
            "Seq.": sequence_by_day_line[key],
            "Reference": ref_id,
            "Family": family,
            "Master boxes": s.get("master_boxes"),
            "Production time (min)": round(production_time, 2),
            "Setup time (min)": round(setup_time, 2),
            "Total occupied (min)": round(production_time + setup_time, 2),
            "Operators": operators,
            "Delivery date": get_delivery_date(instance, s),
            "Delivery day": delivery_day,
            "Delay (days)": delay,
            "Status": "Scheduled",
            "Original order": original_index + 1,
        })

    plan_df = pd.DataFrame(plan_rows)

    if not plan_df.empty:
        plan_df = plan_df.sort_values(["Day", "Line", "Seq."])

    return plan_df


def build_postponed_df(instance, best_solution):
    rows = []

    for original_index, item in enumerate(best_solution):
        if not item.get("postponed"):
            continue

        rows.append({
            "Original order": original_index + 1,
            "Reference": item.get("ref_id"),
            "Master boxes": item.get("master_boxes"),
            "Delivery date": get_delivery_date(instance, item),
            "Delivery day": item.get("delivery_date"),
            "Status": "Postponed",
        })

    postponed_df = pd.DataFrame(rows)

    if not postponed_df.empty:
        postponed_df = postponed_df.sort_values([
            "Delivery day",
            "Reference",
            "Original order",
        ])

    return postponed_df


def build_compact_schedule_df(instance, plan_df, best_metrics):
    compact_rows = []

    for day in range(1, instance["n_days"] + 1):
        day_df = plan_df[plan_df["Day"] == day] if not plan_df.empty else pd.DataFrame()
        row = {"Production date": get_production_date(instance, day)}
        has_capacity_excess = False

        for line in ["L1", "L2"]:
            line_df = day_df[day_df["Line"] == line] if not day_df.empty else pd.DataFrame()
            cap_key = (day, line)
            capacity_excess = best_metrics["capacity_excess_by_day_line"].get(cap_key, 0)

            if line_df.empty:
                row[f"{line} sequence"] = "-"
                row[f"{line} occupied"] = "0 min"
                row[f"{line} excess"] = "0 min"
            else:
                row[f"{line} sequence"] = " -> ".join(
                    line_df["Reference"].astype(str).tolist()
                )
                occupied = (
                    line_df["Production time (min)"].sum()
                    + line_df["Setup time (min)"].sum()
                )
                row[f"{line} occupied"] = f"{occupied:.1f} min"
                row[f"{line} excess"] = f"{capacity_excess:.1f} min"

            if capacity_excess > 0:
                has_capacity_excess = True

        total_operators = best_metrics["operators_required_by_day"].get(day, 0)
        operator_excess = best_metrics["operator_excess_by_day"].get(day, 0)

        row["Operators"] = total_operators
        row["Operator excess"] = operator_excess

        if has_capacity_excess or operator_excess > 0:
            row["Status"] = "Overloaded"
        elif total_operators >= 0.9 * best_metrics["standard_operators"]:
            row["Status"] = "Near limit"
        else:
            row["Status"] = "OK"

        compact_rows.append(row)

    return pd.DataFrame(compact_rows)


def build_capacity_df(instance, best_metrics):
    rows = []

    for day in range(1, instance["n_days"] + 1):
        for line in instance["final_lines"]:
            key = (day, line)
            production_time = best_metrics["production_time_by_day_line"].get(key, 0)
            setup_time = best_metrics["setup_time_by_day_line"].get(key, 0)
            occupied = production_time + setup_time
            capacity_excess = best_metrics["capacity_excess_by_day_line"].get(key, 0)
            available_time = instance["available_line_time_min"]
            utilization = occupied / available_time * 100 if available_time > 0 else 0

            rows.append({
                "Production date": get_production_date(instance, day),
                "Day": day,
                "Line": line,
                "Production time (min)": round(production_time, 2),
                "Setup time (min)": round(setup_time, 2),
                "Occupied time (min)": round(occupied, 2),
                "Available time (min)": available_time,
                "Capacity excess (min)": round(capacity_excess, 2),
                "Utilization (%)": round(utilization, 1),
            })

    return pd.DataFrame(rows)


def build_daily_operator_df(instance, best_metrics):
    rows = []

    for day in range(1, instance["n_days"] + 1):
        required = best_metrics["operators_required_by_day"].get(day, 0)
        excess = best_metrics["operator_excess_by_day"].get(day, 0)

        rows.append({
            "Production date": get_production_date(instance, day),
            "Day": day,
            "Required operators": required,
            "Standard operators": best_metrics["standard_operators"],
            "Operator excess": excess,
        })

    return pd.DataFrame(rows)


def build_hourly_operator_df(instance, best_metrics):
    rows = []
    usage = best_metrics.get("operators_required_by_time", {})
    excess = best_metrics.get("operator_excess_by_time", {})
    standard = best_metrics["standard_operators"]

    for (day, bucket), required in sorted(usage.items()):
        start_min = bucket * TIME_BUCKET_MIN
        end_min = start_min + TIME_BUCKET_MIN

        rows.append({
            "Production date": get_production_date(instance, day),
            "Day": day,
            "Time interval": f"{format_time_from_minutes(start_min)}-{format_time_from_minutes(end_min)}",
            "Required operators": required,
            "Standard operators": standard,
            "Operator excess": excess.get((day, bucket), 0),
        })

    return pd.DataFrame(rows)


def build_time_slot_activity_df(instance, best_metrics):
    operations = best_metrics.get("time_operations", [])
    usage = best_metrics.get("operators_required_by_time", {})
    excess = best_metrics.get("operator_excess_by_time", {})
    standard = best_metrics["standard_operators"]

    if operations:
        max_end = max(op.get("end") or 0 for op in operations)
        last_bucket = max(
            int((max_end - 1) // TIME_BUCKET_MIN),
            int((17 * 60) // TIME_BUCKET_MIN)
        )
    else:
        last_bucket = int((17 * 60) // TIME_BUCKET_MIN)

    first_bucket = int((8 * 60) // TIME_BUCKET_MIN)
    rows = []

    for day in range(1, instance["n_days"] + 1):
        for bucket in range(first_bucket, last_bucket + 1):
            start_min = bucket * TIME_BUCKET_MIN
            end_min = start_min + TIME_BUCKET_MIN

            line_data = {
                "L1": {
                    "references": [],
                    "activities": [],
                    "operators": 0,
                },
                "L2": {
                    "references": [],
                    "activities": [],
                    "operators": 0,
                },
            }

            for op in operations:
                if op.get("day") != day:
                    continue

                line = op.get("line")
                if line not in line_data:
                    continue

                op_start = op.get("start")
                op_end = op.get("end")

                if op_start is None or op_end is None:
                    continue

                overlaps_slot = op_start < end_min and op_end > start_min

                if overlaps_slot:
                    ref_id = str(op.get("ref_id"))
                    operation = str(op.get("operation"))
                    activity = f"{ref_id} ({operation})"

                    if ref_id not in line_data[line]["references"]:
                        line_data[line]["references"].append(ref_id)

                    if activity not in line_data[line]["activities"]:
                        line_data[line]["activities"].append(activity)

                    line_data[line]["operators"] += op.get("operators") or 0

            rows.append({
                "Production date": get_production_date(instance, day),
                "Day": day,
                "Time slot": f"{format_time_from_minutes(start_min)}-{format_time_from_minutes(end_min)}",
                "L1 references": " | ".join(line_data["L1"]["references"]),
                "L1 activity": " | ".join(line_data["L1"]["activities"]),
                "L1 operators": line_data["L1"]["operators"],
                "L2 references": " | ".join(line_data["L2"]["references"]),
                "L2 activity": " | ".join(line_data["L2"]["activities"]),
                "L2 operators": line_data["L2"]["operators"],
                "Total operators used": usage.get((day, bucket), 0),
                "Standard operators": standard,
                "Operator excess": excess.get((day, bucket), 0),
            })

    return pd.DataFrame(rows)


def build_operations_df(instance, best_metrics):
    rows = []

    for op in best_metrics.get("time_operations", []):
        rows.append({
            "Production date": get_production_date(instance, op.get("day")),
            "Day": op.get("day"),
            "Line": op.get("line"),
            "Reference": op.get("ref_id"),
            "Operation": op.get("operation"),
            "Start": format_time_from_minutes(op.get("start")),
            "End": format_time_from_minutes(op.get("end")),
            "Operators": op.get("operators"),
        })

    operations_df = pd.DataFrame(rows)

    if not operations_df.empty:
        operations_df = operations_df.sort_values(["Day", "Line", "Start"])

    return operations_df


def build_penalty_df(best_metrics):
    rows = [
        {
            "Component": "Invalid assignments",
            "Penalty": best_metrics.get("invalid_assignments", 0) * INVALID_LINE_PENALTY,
        },
        {
            "Component": "Capacity excess",
            "Penalty": best_metrics.get("total_capacity_excess", 0) * CAPACITY_PENALTY,
        },
        {
            "Component": "Delay",
            "Penalty": best_metrics.get("delay_days_total", 0) * DELAY_PENALTY,
        },
        {
            "Component": "Postponement",
            "Penalty": best_metrics.get("postponement_penalty", 0),
        },
        {
            "Component": "Hourly operators",
            "Penalty": best_metrics.get("hourly_operator_penalty", 0),
        },
        {
            "Component": "Setup",
            "Penalty": best_metrics.get("setup_penalty", 0),
        },
    ]

    return pd.DataFrame(rows)


def highlight_status(row):
    status = row.get("Status")

    if status == "Overloaded":
        return ["background-color: #4c1d1d; color: #fecaca"] * len(row)

    if status == "Near limit":
        return ["background-color: #3f2f12; color: #fde68a"] * len(row)

    if status == "OK":
        return ["background-color: #123524; color: #bbf7d0"] * len(row)

    return ["background-color: #111827; color: #e5e7eb"] * len(row)


st.set_page_config(
    page_title="Production Sequencing Dashboard",
    layout="wide",
)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
default_excel_path = os.path.join(project_root, "Inputs_Doceleia.xlsx")

POPULATION_SIZE = 100
GENERATIONS = 100
MUTATION_RATE = 0.10
ELITE_SIZE = 5
TOURNAMENT_SIZE = 3
RANDOM_SEED = 42

instance = load_real_instance(default_excel_path)
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
    <div style="
        background-color:#111827;
        padding:18px 24px;
        border-radius:8px;
        margin-bottom:20px;
        border:1px solid #334155;
    ">
        <div style="color:#9ca3af;font-size:14px;font-weight:500;margin-bottom:4px;">
            Kaizen Institute
        </div>
        <div style="color:#f9fafb;font-size:34px;font-weight:800;letter-spacing:0;">
            PRODUCTION SCHEDULING - X COMPANY
        </div>
        <div style="color:#cbd5e1;font-size:18px;font-weight:500;margin-top:6px;">
            Planning month: {planning_month}
        </div>
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
    hourly_operator_df = build_hourly_operator_df(instance, best_metrics)
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
        .mark_bar(color="#ef4444")
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
    capacity_chart = (
        alt.Chart(capacity_df)
        .mark_bar()
        .encode(
            x=alt.X("Production date:N", title="Production date"),
            xOffset=alt.XOffset("Line:N"),
            y=alt.Y("Utilization (%):Q", title="Utilization (%)"),
            color=alt.Color("Line:N", scale=alt.Scale(range=LINE_COLORS)),
            tooltip=[
                "Production date",
                "Line",
                "Production time (min)",
                "Setup time (min)",
                "Occupied time (min)",
                "Available time (min)",
                "Capacity excess (min)",
                "Utilization (%)",
            ],
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
                    "Total operators used:Q",
                    title="Total operators used",
                    scale=alt.Scale(scheme="reds"),
                ),
                tooltip=[
                    "Production date",
                    "Time slot",
                    "L1 activity",
                    "L2 activity",
                    "L1 references",
                    "L1 activity",
                    "L1 operators",
                    "L2 references",
                    "L2 activity",
                    "L2 operators",
                    "Total operators used",
                    "Standard operators",
                    "Operator excess",
                ],
            )
            .properties(height=max(320, 24 * time_slot_df["Production date"].nunique()))
        )
        st.altair_chart(hourly_heatmap, width="stretch")

        st.markdown("Complete hourly table")
        st.dataframe(
            time_slot_df,
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

    st.subheader("Setup Time by Day and Line")
    setup_rows = []

    for key, setup_time in best_metrics["setup_time_by_day_line"].items():
        day, line = key
        setup_rows.append({
            "Production date": get_production_date(instance, day),
            "Line": line,
            "Setup time (min)": round(setup_time, 2),
        })

    setup_df = pd.DataFrame(setup_rows)

    if setup_df.empty:
        st.info("No setup time registered.")
    else:
        setup_chart = (
            alt.Chart(setup_df)
            .mark_bar()
            .encode(
                x=alt.X("Production date:N", title="Production date"),
                xOffset=alt.XOffset("Line:N"),
                y=alt.Y("Setup time (min):Q", title="Setup time (min)"),
                color=alt.Color("Line:N", scale=alt.Scale(range=LINE_COLORS)),
                tooltip=["Production date", "Line", "Setup time (min)"],
            )
            .properties(height=320)
        )
        st.altair_chart(setup_chart, width="stretch")
        st.dataframe(setup_df, width="stretch", hide_index=True)
else:
    st.info("Press Run Genetic Algorithm to generate the production schedule.")
