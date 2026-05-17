import streamlit as st
import pandas as pd

from generate_instance import load_real_instance
from geneticalgorithm import run_genetic_algorithm

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


st.set_page_config(
    page_title="Production Sequencing Dashboard",
    layout="wide"
)

st.title("Production Sequencing Dashboard")

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

    st.subheader("Solution Metrics")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Total penalty",
        f"{best_metrics['total_penalty']:.2f}"
    )

    col2.metric(
        "Delay",
        f"{best_metrics['delay_days_total']} days"
    )

    col3.metric(
        "Capacity excess",
        f"{best_metrics['total_capacity_excess']:.2f} min"
    )

    col4.metric(
        "Operator excess",
        f"{best_metrics['total_operator_excess']:.0f}"
    )

    col5.metric(
        "Setup time",
        f"{best_metrics['setup_total_min']:.0f} min"
    )

    st.subheader("Best Production Plan")

    plan_rows = []

    for item in best_solution:
        production_date = get_production_date(instance, item["day"])

        delivery_date = item.get("delivery_calendar_date")

        if delivery_date is None:
            order_id = item.get("order_id")
            if order_id is not None:
                delivery_date = instance["demand"][order_id].get(
                    "delivery_calendar_date"
                )

        plan_rows.append({
            "Production date": production_date,
            "Line": item["line"],
            "Reference": item["ref_id"],
            "Master boxes": item["master_boxes"],
            "Delivery date": format_date(delivery_date),
            "Priority": item["priority"],
        })

    plan_df = pd.DataFrame(plan_rows)

    st.dataframe(
        plan_df,
        use_container_width=True
    )
