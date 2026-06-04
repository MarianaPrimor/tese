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


LINE_COLORS = ["#153e7e", "#b6003b"]
OVERLOAD_BG = "#f8d7da"
OVERLOAD_TEXT = "#7f1d1d"
NEUTRAL_BG = "#ffffff"
NEAR_LIMIT_BG = "#fff3cd"

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
            refs = " -> ".join(
                f"{row['Reference']} ({row['Master boxes']} boxes)"
                for _, row in line_df.iterrows()
            )
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
        {"Component": "Capacity", "Penalty": best_metrics.get("total_capacity_excess", 0)},
        {"Component": "Delay", "Penalty": best_metrics.get("delay_days_total", 0)},
        {"Component": "Postponement", "Penalty": best_metrics.get("postponement_penalty", 0)},
        {"Component": "Hourly operators", "Penalty": best_metrics.get("hourly_operator_penalty", 0)},
        {"Component": "Setup", "Penalty": best_metrics.get("setup_penalty", 0)},
        {"Component": "Economic reward", "Penalty": -best_metrics.get("economic_value_reward", 0)},
    ]

    return pd.DataFrame(rows)


def highlight_status(row):
    status = row.get("Status")
    status = row.get("Estado", status)

    if status in ["Overloaded", "Sobrecarga"]:
        return [f"background-color: {OVERLOAD_BG}; color: {OVERLOAD_TEXT}"] * len(row)

    if status in ["Near limit", "Perto do limite"]:
        return [f"background-color: {NEAR_LIMIT_BG}; color: #7a5a00"] * len(row)

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


st.set_page_config(
    page_title="Planeamento de Produção - Doceleia",
    layout="wide",
)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
default_excel_path = os.path.join(project_root, "Inputs_Doceleia.xlsx")

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
    textarea,
    input {
        background-color: #ffffff !important;
        color: #172033 !important;
        border: 1px solid #b8c7d9 !important;
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
    div[data-testid="stDataFrame"],
    div[data-testid="stDataFrame"] div,
    div[data-testid="stDataFrame"] table,
    div[data-testid="stTable"],
    div[data-testid="stTable"] table {
        background-color: #ffffff !important;
        color: #172033 !important;
    }
    div[data-testid="stDataFrame"] th,
    div[data-testid="stDataFrame"] td,
    div[data-testid="stTable"] th,
    div[data-testid="stTable"] td {
        color: #172033 !important;
        background-color: #ffffff !important;
        border-color: #d9e0ea !important;
    }
    div[data-testid="stDataFrame"] thead tr th,
    div[data-testid="stTable"] thead tr th {
        background-color: #153e7e !important;
        color: #ffffff !important;
    }
    div[data-testid="stDataFrame"] [role="grid"],
    div[data-testid="stDataFrame"] [role="columnheader"],
    div[data-testid="stDataFrame"] [role="gridcell"] {
        background-color: #ffffff !important;
        color: #172033 !important;
    }
    div[data-testid="stDataFrame"] [role="columnheader"] {
        background-color: #153e7e !important;
        color: #ffffff !important;
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

run_button = st.button("Gerar plano de produção", width="content")

if not run_button:
    st.info("Pressione o botão para gerar o plano de produção.")
    st.stop()

excel_source = uploaded_excel if uploaded_excel is not None else excel_path

try:
    with st.spinner("A carregar dados e a correr o algoritmo genético..."):
        instance = load_real_instance(excel_source)
        planning_month = get_planning_month(instance)
        best_solution, best_metrics = run_genetic_algorithm(
            instance,
            population_size=POPULATION_SIZE,
            generations=GENERATIONS,
            mutation_rate=MUTATION_RATE,
            elite_size=ELITE_SIZE,
            tournament_size=TOURNAMENT_SIZE,
            seed=RANDOM_SEED,
        )
except Exception as exc:
    st.error(f"Não foi possível gerar o plano: {exc}")
    st.stop()

st.success(f"Plano gerado com sucesso para {planning_month}.")

plan_df = build_plan_df(instance, best_solution)
compact_df = build_compact_schedule_df(instance, plan_df, best_metrics)
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

st.subheader("Sequência diária de produção")
sequencing_df = compact_df.rename(columns={
    "Production date": "Data de produção",
    "Day": "Dia",
    "L1 sequence": "Sequência L1",
    "L1 production": "Produção L1",
    "L1 setup": "Setup L1",
    "L1 excess": "Excesso L1",
    "L2 sequence": "Sequência L2",
    "L2 production": "Produção L2",
    "L2 setup": "Setup L2",
    "L2 excess": "Excesso L2",
    "Status": "Estado",
})
sequencing_df["Estado"] = sequencing_df["Estado"].replace({
    "Overloaded": "Sobrecarga",
    "Near limit": "Perto do limite",
    "OK": "OK",
})
st.dataframe(
    sequencing_df.style.apply(highlight_status, axis=1),
    width="stretch",
    hide_index=True,
    height=420,
)

st.subheader("Quantidade e valor por produto, dia e linha")
product_matrix_display_df = product_matrix_df.rename(columns={
    "Production date": "Data de produção",
    "Day": "Dia",
    "Line": "Linha",
})
st.dataframe(
    light_table_style(product_matrix_display_df),
    width="stretch",
    hide_index=True,
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
            x=alt.X("Time slot:N", title="Horário"),
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
    st.dataframe(
        operators_df.style.apply(highlight_status, axis=1),
        width="stretch",
        hide_index=True,
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
        "Production time (min)": "Tempo de produção (min)",
        "Setup time (min)": "Tempo de setup (min)",
        "Occupied time (min)": "Tempo ocupado (min)",
        "Available time (min)": "Tempo disponível (min)",
        "Capacity excess (min)": "Excesso de capacidade (min)",
        "Utilization (%)": "Utilização (%)",
    })
    st.dataframe(
        light_table_style(capacity_display_df),
        width="stretch",
        hide_index=True,
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
st.dataframe(
    light_table_style(plan_display_df),
    width="stretch",
    hide_index=True,
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
    st.dataframe(
        light_table_style(operations_display_df),
        width="stretch",
        hide_index=True,
        height=520,
    )
