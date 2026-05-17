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
)


def format_date(value):
    if value is None:
        return ""

    if hasattr(value, "strftime"):
        return value.strftime("%d/%m/%Y")

    return str(value)


def get_production_date(instance, day):
    working_days = instance.get("working_days", [])

    if working_days and 1 <= day <= len(working_days):
        return format_date(working_days[day - 1])

    return f"Day {day}"


def get_delivery_date(instance, solution_item):
    delivery_date = solution_item.get("delivery_calendar_date")

    if delivery_date is not None:
        return format_date(delivery_date)

    order_id = solution_item.get("order_id")

    if order_id is not None:
        demand_order = instance["demand"][order_id]
        delivery_date = demand_order.get("delivery_calendar_date")

        if delivery_date is not None:
            return format_date(delivery_date)

    return str(solution_item.get("delivery_date", ""))


def build_plan_df(instance, best_solution):
    refs_by_id = create_refs_by_id(instance)
    plan_rows = []
    last_family_by_day_line = {}

    sorted_solution = sorted(
        enumerate(best_solution),
        key=lambda x: (
            x[1].get("day"),
            x[1].get("line") or "INVALID",
            x[0],
        )
    )

    sequence_by_day_line = {}

    for original_index, s in sorted_solution:
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
                s.get("master_boxes")
            ) or 0

            operators = get_required_operators(ref, line)

            setup_time = get_setup(
                instance,
                last_family_by_day_line.get(key),
                ref["family"]
            )

            last_family_by_day_line[key] = ref["family"]
            family = ref["family"]
        else:
            production_time = 0
            operators = 0
            setup_time = 0
            family = ""

        delay = max(0, day - s.get("delivery_date", day))

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
            "Operators": operators,
            "Delivery date": get_delivery_date(instance, s),
            "Delay (days)": delay,
            "Priority": s.get("priority"),
            "Original order": original_index + 1,
        })

    plan_df = pd.DataFrame(plan_rows)

    if not plan_df.empty:
        plan_df = plan_df.sort_values(["Day", "Line", "Seq."])

    return plan_df


def build_compact_schedule_df(instance, plan_df, best_metrics):
    compact_rows = []

    if plan_df.empty:
        return pd.DataFrame()

    for day in sorted(plan_df["Day"].unique()):
        day_df = plan_df[plan_df["Day"] == day]

        row = {
            "Production date": get_production_date(instance, day)
        }

        has_capacity_excess = False

        for line in ["L1", "L2"]:
            line_df = day_df[day_df["Line"] == line]

            if line_df.empty:
                row[f"{line} sequence"] = "-"
                row[f"{line} load"] = "0 min"
                row[f"{line} setup"] = "0 min"
            else:
                row[f"{line} sequence"] = " → ".join(
                    line_df["Reference"].astype(str).tolist()
                )

                load = line_df["Production time (min)"].sum()
                setup = line_df["Setup time (min)"].sum()

                row[f"{line} load"] = f"{load:.1f} min"
                row[f"{line} setup"] = f"{setup:.1f} min"

                if load > instance["available_line_time_min"]:
                    has_capacity_excess = True

        total_operators = best_metrics["operators_required_by_day"].get(
            day,
            0
        )

        operator_excess = best_metrics["operator_excess_by_day"].get(
            day,
            0
        )

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


def highlight_status(row):
    if row["Status"] == "Overloaded":
        return [
            "background-color: #4c1d1d; color: #fecaca"
        ] * len(row)

    if row["Status"] == "Near limit":
        return [
            "background-color: #3f2f12; color: #fde68a"
        ] * len(row)

    if row["Status"] == "OK":
        return [
            "background-color: #123524; color: #bbf7d0"
        ] * len(row)

    return [
        "background-color: #111827; color: #e5e7eb"
    ] * len(row)



st.set_page_config(
    page_title="Production Sequencing Dashboard",
    layout="wide"
)

st.markdown(
    """
    <div style="
        background-color:#1f2933;
        padding:18px 24px;
        border-radius:8px;
        margin-bottom:24px;
        border:1px solid #334155;
    ">
        <div style="
            color:#9ca3af;
            font-size:14px;
            font-weight:500;
            margin-bottom:4px;
        ">
            Kaizen Institute
        </div>
        <div style="
            color:#f9fafb;
            font-size:32px;
            font-weight:700;
        ">
            Production Sequencing Dashboard
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


excel_path = st.text_input(
    "Excel file path",
    "../Inputs_Doceleia.xlsx"
)

population_size = st.number_input(
    "Population size",
    min_value=10,
    max_value=500,
    value=100,
    step=10
)

generations = st.number_input(
    "Generations",
    min_value=1,
    max_value=500,
    value=100,
    step=10
)

mutation_rate = st.slider(
    "Mutation rate",
    min_value=0.0,
    max_value=1.0,
    value=0.10,
    step=0.01
)

if st.button("Run Genetic Algorithm"):
    with st.spinner("Loading data and running genetic algorithm..."):
        instance = load_real_instance(excel_path)

        best_solution, best_metrics = run_genetic_algorithm(
            instance,
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            elite_size=5,
            tournament_size=3,
            seed=42,
        )

    st.success("Genetic algorithm finished.")

    plan_df = build_plan_df(instance, best_solution)
    compact_df = build_compact_schedule_df(
        instance,
        plan_df,
        best_metrics
    )

    st.subheader("Solution Metrics")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total penalty", f"{best_metrics['total_penalty']:.2f}")
    col2.metric("Delay", f"{best_metrics['delay_days_total']} days")
    col3.metric(
        "Capacity excess",
        f"{best_metrics['total_capacity_excess']:.2f} min"
    )
    col4.metric(
        "Operator excess",
        f"{best_metrics['total_operator_excess']:.0f}"
    )
    col5.metric("Setup time", f"{best_metrics['setup_total_min']:.0f} min")

    st.subheader("Daily Production Schedule")

    st.dataframe(
        compact_df.style.apply(highlight_status, axis=1),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Detailed Production Plan")

    st.dataframe(
        plan_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Capacity Usage")

    capacity_rows = []

    for key, production_time in best_metrics["production_time_by_day_line"].items():
        day, line = key

        capacity_excess = best_metrics["capacity_excess_by_day_line"].get(
            key,
            0
        )

        available_time = instance["available_line_time_min"]

        utilization = (
            production_time / available_time * 100
            if available_time > 0
            else 0
        )

        capacity_rows.append({
            "Production date": get_production_date(instance, day),
            "Line": line,
            "Production time (min)": round(production_time, 2),
            "Available time (min)": available_time,
            "Capacity excess (min)": round(capacity_excess, 2),
            "Utilization (%)": round(utilization, 1),
        })

    capacity_df = pd.DataFrame(capacity_rows)

    st.dataframe(
        capacity_df,
        use_container_width=True,
        hide_index=True
    )

    capacity_chart = (
        alt.Chart(capacity_df)
        .mark_bar()
        .encode(
            x=alt.X("Production date:N", title="Production date"),
            xOffset=alt.XOffset("Line:N"),
            y=alt.Y("Utilization (%):Q", title="Utilization (%)"),
            color=alt.Color("Line:N", title="Line"),
            tooltip=[
                "Production date",
                "Line",
                "Production time (min)",
                "Available time (min)",
                "Capacity excess (min)",
                "Utilization (%)",
            ],
        )
        .properties(height=350)
    )

    st.altair_chart(
        capacity_chart,
        use_container_width=True
    )

    st.subheader("Operators Required")

    operator_rows = []

    for day, required_operators in best_metrics["operators_required_by_day"].items():
        operator_excess = best_metrics["operator_excess_by_day"].get(day, 0)

        operator_rows.append({
            "Production date": get_production_date(instance, day),
            "Required operators": required_operators,
            "Standard operators": best_metrics["standard_operators"],
            "Operator excess": operator_excess,
        })

    operators_df = pd.DataFrame(operator_rows)

    st.dataframe(
        operators_df,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        operators_df,
        x="Production date",
        y=["Required operators", "Standard operators"]
    )

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

    st.dataframe(
        setup_df,
        use_container_width=True,
        hide_index=True
    )

    setup_chart = (
        alt.Chart(setup_df)
        .mark_bar()
        .encode(
            x=alt.X("Production date:N", title="Production date"),
            xOffset=alt.XOffset("Line:N"),
            y=alt.Y("Setup time (min):Q", title="Setup time (min)"),
            color=alt.Color("Line:N", title="Line"),
            tooltip=[
                "Production date",
                "Line",
                "Setup time (min)",
            ],
        )
        .properties(height=350)
    )

    st.altair_chart(
        setup_chart,
        use_container_width=True
    )
