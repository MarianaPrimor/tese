import os
from copy import deepcopy
from datetime import date, time, timedelta

import altair as alt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from generate_instance import load_real_instance
from geneticalgorithm import run_genetic_algorithm
from evaluator import (
    DEFAULT_NORMALISED_WEIGHTS,
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

POPULATION_SIZE = 179
GENERATIONS = 200
MUTATION_RATE = 0.05280868822211533
ELITE_SIZE = 5
TOURNAMENT_SIZE = 3
STAGNATION_K = 35
RANDOM_SEED = 42
DEFAULT_OPERATORS = 20
LUNCH_BREAK_MIN = 30
SHIFT_GROSS_CAPACITY_MIN = 8 * 60


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

        selected_duration = max(0, end_min - start_min)
        gross_capacity = selected_duration * n_shifts

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
    instance["standard_operators"] = (
        max(operators_by_day.values())
        if operators_by_day
        else DEFAULT_OPERATORS
    )
    instance["daily_shift_start_min"] = daily_shift_start_min
    instance["daily_shift_end_min"] = daily_shift_end_min
    instance["daily_capacity_min"] = daily_capacity_min
    instance["daily_shifts"] = daily_shifts

    if daily_capacity_min:
        instance["available_line_time_min"] = min(daily_capacity_min.values())

    return instance


def get_production_date(instance, day):
    if day is None:
        return "Adiado"

    working_days = instance.get("working_days", [])

    if working_days and 1 <= day <= len(working_days):
        return format_date(working_days[day - 1])

    return f"Dia {day}"


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


def build_simple_daily_plan_df(instance, plan_df):
    scheduled_df = plan_df[
        plan_df["Status"] == "Scheduled"
    ].copy()

    if scheduled_df.empty:
        return pd.DataFrame()

    rows = []
    scheduled_df = scheduled_df.sort_values(["Day", "Line", "Seq."])

    for (day, line), group_df in scheduled_df.groupby(["Day", "Line"], sort=True):
        production_total = 0
        setup_total = 0

        for _, row in group_df.iterrows():
            production_time = row.get("Production time (min)", 0) or 0
            setup_time = row.get("Setup time (min)", 0) or 0
            production_total += production_time
            setup_total += setup_time

            rows.append({
                "Dia": int(day),
                "Produto": row.get("Reference", ""),
                "Quantidade": row.get("Master boxes", 0),
                "Linha": line,
                "Tempo de produção (h)": round(production_time / 60, 2),
                "Tempo de setup (h)": round(setup_time / 60, 2),
                "Limpeza final (h)": "",
                "Tempo total (h)": round((production_time + setup_time) / 60, 2),
                "_tipo": "produto",
                "_estado": "",
            })

        available_time = get_available_line_time_for_day(instance, day)
        tolerance = get_capacity_tolerance_for_day(instance, day)
        cleaning_time = instance.get("end_of_day_cleaning_time_min", 0)
        occupied_without_cleaning = production_total + setup_total
        occupied_with_cleaning = occupied_without_cleaning + cleaning_time
        excess_over_tolerance = (
            occupied_without_cleaning
            - available_time
            - tolerance
        )
        total_status = (
            "warning"
            if excess_over_tolerance > 0
            else "ok"
        )

        rows.append({
            "Dia": int(day),
            "Produto": f"TOTAL DIA {int(day)} - {line}",
            "Quantidade": "",
            "Linha": line,
            "Tempo de produção (h)": round(production_total / 60, 2),
            "Tempo de setup (h)": round(setup_total / 60, 2),
            "Limpeza final (h)": round(cleaning_time / 60, 2),
            "Tempo total (h)": round(occupied_with_cleaning / 60, 2),
            "_tipo": "total",
            "_estado": total_status,
        })

    return pd.DataFrame(rows)


def build_postponed_orders_df(plan_df):
    postponed_df = plan_df[
        plan_df["Status"] == "Postponed"
    ].copy()

    if postponed_df.empty:
        return pd.DataFrame()

    postponed_df = postponed_df.sort_values(["Delivery day", "Reference"])

    return postponed_df.rename(columns={
        "Reference": "Produto",
        "Master boxes": "Quantidade",
        "Kg": "Kg",
        "Economic value": "Valor económico",
        "Delivery date": "Data de entrega",
        "Delivery day": "Dia de entrega",
    })[[
        "Produto",
        "Quantidade",
        "Kg",
        "Valor económico",
        "Data de entrega",
        "Dia de entrega",
    ]]


def render_simple_daily_plan_table(simple_plan_df, key, height=520):
    if simple_plan_df.empty:
        st.info("Não existem ordens planeadas.")
        return

    display_df = simple_plan_df.drop(columns=["_tipo", "_estado"])

    def style_rows(row):
        source_row = simple_plan_df.loc[row.name]

        if source_row["_tipo"] != "total":
            return [""] * len(row)

        if source_row["_estado"] == "warning":
            return [
                f"background-color: {NEAR_LIMIT_BG}; font-weight: 700;"
            ] * len(row)

        return [
            f"background-color: {OK_BG}; color: {OK_TEXT}; font-weight: 700;"
        ] * len(row)

    styled_df = (
        display_df.style
        .apply(style_rows, axis=1)
        .format({
            "Tempo de produção (h)": "{:.2f}",
            "Tempo de setup (h)": "{:.2f}",
            "Tempo total (h)": "{:.2f}",
        })
    )

    st.dataframe(
        styled_df,
        width="stretch",
        hide_index=True,
        height=height,
    )


def render_postponed_orders_table(postponed_df, key, height=260):
    if postponed_df.empty:
        st.info("Não existem pedidos adiados.")
        return

    styled_df = postponed_df.style.format({
        "Kg": "{:,.2f}",
        "Valor económico": "€ {:,.2f}",
    })

    st.dataframe(
        styled_df,
        width="stretch",
        hide_index=True,
        height=height,
    )


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
            "Aptidão normalizada",
            ga_metrics.get("normalised_fitness", 0),
            manual_metrics.get("normalised_fitness", 0),
            "min",
        ),
        (
            "Excesso de capacidade (min)",
            ga_metrics.get("total_capacity_excess", 0),
            manual_metrics.get("total_capacity_excess", 0),
            "min",
        ),
        (
            "Tempo de preparação (min)",
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


def build_capacity_what_if_df(instance):
    rows = []

    for day in range(1, instance["n_days"] + 1):
        total_operators = int(get_standard_operators_for_day(instance, day))

        rows.append({
            "Dia": day,
            "Data": get_production_date(instance, day),
            "Operadores atuais": total_operators,
            "Operadores no cenário": total_operators,
        })

    return pd.DataFrame(rows)


def apply_capacity_what_if(instance, edited_capacity_df):
    scenario_instance = deepcopy(instance)
    operators_by_day = {}

    for _, row in edited_capacity_df.iterrows():
        day = int(row["Dia"])
        operators_by_day[day] = int(max(0, row.get("Operadores no cenário", 0) or 0))

    scenario_instance["standard_operators_by_day"] = operators_by_day

    if operators_by_day:
        scenario_instance["standard_operators"] = max(operators_by_day.values())

    return scenario_instance


def build_what_if_comparison_df(baseline_metrics, scenario_metrics):
    metric_rows = [
        ("Ordens adiadas", "postponed_orders", "min"),
        ("Caixas adiadas", "postponed_boxes", "min"),
        ("Valor produzido", "scheduled_economic_value", "max"),
        ("Operadores totais usados", "operator_usage_minutes", "max"),
    ]
    rows = []

    for label, key, direction in metric_rows:
        baseline_value = baseline_metrics.get(key, 0)
        scenario_value = scenario_metrics.get(key, 0)
        delta = scenario_value - baseline_value

        if abs(delta) < 1e-9:
            result = "Sem alteração"
        elif (direction == "min" and delta < 0) or (direction == "max" and delta > 0):
            result = "Melhorou"
        else:
            result = "Piorou"

        rows.append({
            "Métrica": label,
            "Baseline": round(baseline_value, 2),
            "Cenário": round(scenario_value, 2),
            "Δ": round(delta, 2),
            "Resultado": result,
        })

    return pd.DataFrame(rows)


def build_baseline_scenario_bar_df(baseline_metrics, scenario_metrics):
    return pd.DataFrame([
        {
            "Cenário": "Baseline",
            "Ordens adiadas": baseline_metrics.get("postponed_orders", 0),
            "Caixas adiadas": baseline_metrics.get("postponed_boxes", 0),
            "Valor produzido": baseline_metrics.get("scheduled_economic_value", 0),
            "Operadores usados": baseline_metrics.get("operator_usage_minutes", 0),
        },
        {
            "Cenário": "Cenário",
            "Ordens adiadas": scenario_metrics.get("postponed_orders", 0),
            "Caixas adiadas": scenario_metrics.get("postponed_boxes", 0),
            "Valor produzido": scenario_metrics.get("scheduled_economic_value", 0),
            "Operadores usados": scenario_metrics.get("operator_usage_minutes", 0),
        },
    ])


def render_baseline_scenario_bar_chart(baseline_metrics, scenario_metrics, title):
    chart_df = build_baseline_scenario_bar_df(baseline_metrics, scenario_metrics)
    chart_long = chart_df.melt(
        id_vars="Cenário",
        var_name="Métrica",
        value_name="Valor",
    )
    fig = px.bar(
        chart_long,
        x="Métrica",
        y="Valor",
        color="Cenário",
        barmode="group",
        color_discrete_map={
            "Baseline": "#153e7e",
            "Cenário": "#b6003b",
        },
        title=title,
    )
    st.plotly_chart(fig, width="stretch")


def style_what_if_delta(row):
    result = row.get("Resultado")
    style = [""] * len(row)

    if "Δ" not in row.index:
        return style

    delta_index = list(row.index).index("Δ")

    if result == "Melhorou":
        style[delta_index] = (
            f"background-color: {OK_BG}; color: {OK_TEXT}; font-weight: 700"
        )
    elif result == "Piorou":
        style[delta_index] = (
            f"background-color: {OVERLOAD_BG}; color: {OVERLOAD_TEXT}; font-weight: 700"
        )

    return style


def render_capacity_what_if_section(instance, baseline_solution, baseline_metrics):
    st.subheader("What-If de Capacidade")
    st.caption("Edite a disponibilidade de operadores por dia.")

    default_capacity_df = build_capacity_what_if_df(instance)
    max_operator_option = max(
        int(default_capacity_df["Operadores atuais"].max()) + 10,
        30,
    )
    edited_capacity_df = st.data_editor(
        default_capacity_df,
        column_config={
            "Dia": st.column_config.NumberColumn(format="%d"),
            "Data": st.column_config.TextColumn(),
            "Operadores atuais": st.column_config.NumberColumn(format="%d"),
            "Operadores no cenário": st.column_config.SelectboxColumn(
                options=list(range(0, max_operator_option + 1)),
                required=True,
            ),
        },
        disabled=["Dia", "Data", "Operadores atuais"],
        hide_index=True,
        width="stretch",
        height=360,
        key="capacity_what_if_editor",
    )

    if st.button("Simular cenário", type="primary", width="content"):
        scenario_instance = apply_capacity_what_if(instance, edited_capacity_df)

        with st.spinner("A correr o GA para o cenário What-If..."):
            scenario_solution, scenario_metrics, _ = run_genetic_algorithm(
                scenario_instance,
                population_size=POPULATION_SIZE,
                generations=GENERATIONS,
                mutation_rate=MUTATION_RATE,
                elite_size=ELITE_SIZE,
                tournament_size=TOURNAMENT_SIZE,
                stagnation_k=STAGNATION_K,
                seed=RANDOM_SEED,
            )

        scenario_max_values = compute_max_values(scenario_instance)
        scenario_metrics = add_normalised_fitness_metrics(
            evaluate_solution(scenario_solution, scenario_instance),
            scenario_max_values,
        )

        st.session_state["capacity_what_if_instance"] = deepcopy(scenario_instance)
        st.session_state["capacity_what_if_solution"] = deepcopy(scenario_solution)
        st.session_state["capacity_what_if_metrics"] = deepcopy(scenario_metrics)
        st.session_state["capacity_what_if_input"] = edited_capacity_df.copy()

    scenario_metrics = st.session_state.get("capacity_what_if_metrics")

    if scenario_metrics is None:
        st.info("Edite a tabela e clique em Simular cenário para comparar com o baseline.")
        return

    comparison_df = build_what_if_comparison_df(
        baseline_metrics,
        scenario_metrics,
    )

    st.markdown("**Baseline vs. Cenário**")
    st.dataframe(
        comparison_df.style.apply(style_what_if_delta, axis=1),
        width="stretch",
        hide_index=True,
        height=240,
    )
    render_baseline_scenario_bar_chart(
        baseline_metrics,
        scenario_metrics,
        "Comparação Baseline vs. Cenário",
    )


def run_dashboard_ga_scenario(instance, seed=RANDOM_SEED, objective_weights=None):
    solution, _, history = run_genetic_algorithm(
        instance,
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        mutation_rate=MUTATION_RATE,
        elite_size=ELITE_SIZE,
        tournament_size=TOURNAMENT_SIZE,
        stagnation_k=STAGNATION_K,
        seed=seed,
        objective_weights=objective_weights,
    )
    max_values = compute_max_values(instance)
    metrics = add_normalised_fitness_metrics(
        evaluate_solution(solution, instance),
        max_values,
    )
    return solution, metrics, history


def get_ref_abc_class(ref):
    abc_class = str(ref.get("abc_class", "C") or "C").strip().upper()
    return abc_class if abc_class in {"A", "B", "C"} else "C"


def get_order_abc_class(order, refs_by_id):
    ref = refs_by_id.get(str(order.get("ref_id", "")).strip())
    if ref is None:
        return "C"
    return get_ref_abc_class(ref)


def get_order_economic_value(order, refs_by_id):
    ref = refs_by_id.get(str(order.get("ref_id", "")).strip())
    if ref is None:
        return 0
    unit_value = ref.get("economic_value_per_master_box", 0) or 0
    return (order.get("master_boxes", 0) or 0) * unit_value


def get_order_line_label(order, refs_by_id):
    ref = refs_by_id.get(str(order.get("ref_id", "")).strip())

    if ref is None:
        return ""

    valid_lines = valid_lines_for_ref(ref)
    return valid_lines[0] if valid_lines else ""


def get_solution_signature(gene):
    return (
        str(gene.get("ref_id", "")).strip(),
        int(gene.get("master_boxes", 0) or 0),
        gene.get("delivery_date"),
    )


def make_scenario_instance(instance, demand=None, operators_by_day=None):
    scenario_instance = deepcopy(instance)

    if demand is not None:
        scenario_instance["demand"] = deepcopy(demand)

    if operators_by_day is not None:
        scenario_instance["standard_operators_by_day"] = dict(operators_by_day)
        if operators_by_day:
            scenario_instance["standard_operators"] = max(operators_by_day.values())

    return scenario_instance


def build_daily_operator_pool(instance):
    return {
        day: get_standard_operators_for_day(instance, day)
        for day in range(1, instance.get("n_days", 0) + 1)
    }


def compute_overall_capacity_utilization(instance, metrics):
    used = 0
    available = 0

    for day in range(1, instance.get("n_days", 0) + 1):
        day_available = get_available_line_time_for_day(instance, day)

        for line in instance.get("final_lines", []):
            production = metrics.get("production_time_by_day_line", {}).get((day, line), 0)
            setup = metrics.get("setup_time_by_day_line", {}).get((day, line), 0)
            used += production + setup
            available += day_available

    return used / available * 100 if available else 0


def build_automatic_capacity_results(instance, baseline_metrics):
    capacity_df = build_capacity_df(instance, baseline_metrics)
    operator_pool = build_daily_operator_pool(instance)
    baseline_value = baseline_metrics.get("scheduled_economic_value", 0)
    baseline_orders = baseline_metrics.get("postponed_orders", 0)
    baseline_boxes = baseline_metrics.get("postponed_boxes", 0)
    rows = []

    if capacity_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    for day in range(1, instance.get("n_days", 0) + 1):
        day_capacity = capacity_df[capacity_df["Day"] == day]
        if day_capacity.empty:
            line = instance.get("final_lines", ["L1"])[0]
        else:
            line = day_capacity.sort_values("Utilization (%)", ascending=False).iloc[0]["Line"]

        scenario_pool = dict(operator_pool)
        scenario_pool[day] = scenario_pool.get(day, 0) + 1
        scenario_instance = make_scenario_instance(
            instance,
            operators_by_day=scenario_pool,
        )
        _, scenario_metrics, _ = run_dashboard_ga_scenario(
            scenario_instance,
            seed=RANDOM_SEED + day,
        )

        rows.append({
            "Dia": day,
            "Data": get_production_date(instance, day),
            "Linha mais crítica": line,
            "Operadores adicionais": 1,
            "Ordens adiadas": scenario_metrics.get("postponed_orders", 0),
            "Caixas adiadas": scenario_metrics.get("postponed_boxes", 0),
            "Valor económico produzido": scenario_metrics.get("scheduled_economic_value", 0),
            "Ganho económico": (
                scenario_metrics.get("scheduled_economic_value", 0)
                - baseline_value
            ),
            "Redução de ordens adiadas": (
                baseline_orders
                - scenario_metrics.get("postponed_orders", 0)
            ),
            "Redução de caixas adiadas": (
                baseline_boxes
                - scenario_metrics.get("postponed_boxes", 0)
            ),
        })

    individual_df = pd.DataFrame(rows)
    ranked_days = individual_df.sort_values(
        ["Ganho económico", "Redução de caixas adiadas", "Redução de ordens adiadas"],
        ascending=False,
    )["Dia"].tolist()
    cumulative_rows = [{
        "Operadores adicionais": 0,
        "Dias reforçados": "Baseline",
        "Linhas reforçadas": "",
        "Ordens adiadas": baseline_orders,
        "Caixas adiadas": baseline_boxes,
        "Valor económico produzido": baseline_value,
        "Ganho económico": 0,
        "Redução de ordens adiadas": 0,
    }]

    for n_extra in range(1, min(5, len(ranked_days)) + 1):
        scenario_pool = dict(operator_pool)
        selected_days = ranked_days[:n_extra]

        for day in selected_days:
            scenario_pool[day] = scenario_pool.get(day, 0) + 1

        scenario_instance = make_scenario_instance(
            instance,
            operators_by_day=scenario_pool,
        )
        _, scenario_metrics, _ = run_dashboard_ga_scenario(
            scenario_instance,
            seed=RANDOM_SEED + 100 + n_extra,
        )
        selected_lines = [
            str(
                individual_df.loc[
                    individual_df["Dia"] == day,
                    "Linha mais crítica",
                ].iloc[0]
            )
            for day in selected_days
        ]
        cumulative_rows.append({
            "Operadores adicionais": n_extra,
            "Dias reforçados": ", ".join(str(day) for day in selected_days),
            "Linhas reforçadas": ", ".join(selected_lines),
            "Ordens adiadas": scenario_metrics.get("postponed_orders", 0),
            "Caixas adiadas": scenario_metrics.get("postponed_boxes", 0),
            "Valor económico produzido": scenario_metrics.get("scheduled_economic_value", 0),
            "Ganho económico": (
                scenario_metrics.get("scheduled_economic_value", 0)
                - baseline_value
            ),
            "Redução de ordens adiadas": (
                baseline_orders
                - scenario_metrics.get("postponed_orders", 0)
            ),
        })

    return individual_df, pd.DataFrame(cumulative_rows)


def build_capacity_recommendation(cumulative_df):
    if cumulative_df.empty or len(cumulative_df) <= 1:
        return None

    scenario_df = cumulative_df[cumulative_df["Operadores adicionais"] > 0].copy()

    if scenario_df.empty:
        return None

    scenario_df["Pontuação por operador"] = (
        scenario_df["Ganho económico"].clip(lower=0)
        + scenario_df["Redução de ordens adiadas"].clip(lower=0) * 10000
    ) / scenario_df["Operadores adicionais"]
    best_row = scenario_df.sort_values(
        ["Pontuação por operador", "Ganho económico", "Redução de ordens adiadas"],
        ascending=False,
    ).iloc[0]

    day_values = str(best_row["Dias reforçados"]).split(", ")
    line_values = str(best_row["Linhas reforçadas"]).split(", ")
    parts = []

    for day, line in zip(day_values, line_values):
        parts.append(f"+1 operador no dia {day} ({line})")

    return (
        "Configuração recomendada: "
        + ", ".join(parts)
        + ". Impacto: "
        + f"-{int(best_row['Redução de ordens adiadas'])} ordens adiadas, "
        + f"+€{best_row['Ganho económico']:,.0f} de valor económico."
    )


def build_weight_profiles():
    return {
        "Equilibrado": dict(DEFAULT_NORMALISED_WEIGHTS),
        "Financeiro": {
            "postponement": 0.10,
            "economic_value": 0.70,
            "delay": 0.05,
            "setup": 0.10,
            "capacity_utilisation": 0.04,
            "operator_utilisation": 0.01,
        },
        "Serviço": {
            "postponement": 0.15,
            "economic_value": 0.08,
            "delay": 0.65,
            "setup": 0.05,
            "capacity_utilisation": 0.05,
            "operator_utilisation": 0.02,
        },
        "Eficiência": {
            "postponement": 0.10,
            "economic_value": 0.05,
            "delay": 0.05,
            "setup": 0.40,
            "capacity_utilisation": 0.32,
            "operator_utilisation": 0.08,
        },
    }


def build_weight_sensitivity_results(instance):
    rows = []
    seeds = [0, 42, 99]

    for profile_name, weights in build_weight_profiles().items():
        seed_rows = []

        for seed in seeds:
            _, metrics, _ = run_dashboard_ga_scenario(
                instance,
                seed=seed,
                objective_weights=weights,
            )
            seed_rows.append(metrics)

        rows.append({
            "Perfil": profile_name,
            "Ordens adiadas": sum(m.get("postponed_orders", 0) for m in seed_rows) / len(seed_rows),
            "Caixas adiadas": sum(m.get("postponed_boxes", 0) for m in seed_rows) / len(seed_rows),
            "Valor económico produzido": sum(m.get("scheduled_economic_value", 0) for m in seed_rows) / len(seed_rows),
            "Atraso total": sum(m.get("delay_days_total", 0) for m in seed_rows) / len(seed_rows),
            "Setup total": sum(m.get("setup_total_min", 0) for m in seed_rows) / len(seed_rows),
            "Minutos de operadores usados": sum(m.get("operator_usage_minutes", 0) for m in seed_rows) / len(seed_rows),
            "Fitness normalizada": sum(m.get("normalised_fitness", 0) for m in seed_rows) / len(seed_rows),
        })

    return pd.DataFrame(rows)


def style_balanced_profile(row):
    if row.get("Perfil") != "Equilibrado":
        return [""] * len(row)

    return [
        "background-color: #d1e7dd; color: #0f5132; font-weight: 700"
        for _ in row
    ]


def build_portfolio_abc_results(instance):
    refs_by_id = create_refs_by_id(instance)
    scenarios = [
        ("A", {"A"}),
        ("A+B", {"A", "B"}),
        ("A+B+C", {"A", "B", "C"}),
    ]
    rows = []

    for label, classes in scenarios:
        demand_subset = [
            order
            for order in instance.get("demand", [])
            if get_order_abc_class(order, refs_by_id) in classes
        ]
        scenario_instance = make_scenario_instance(instance, demand=demand_subset)
        solution, metrics, _ = run_dashboard_ga_scenario(
            scenario_instance,
            seed=RANDOM_SEED + len(classes),
        )
        postponed_orders = metrics.get("postponed_orders", 0)
        total_orders = len(demand_subset)
        total_value = sum(
            get_order_economic_value(order, refs_by_id)
            for order in demand_subset
        )

        rows.append({
            "Cenário": label,
            "Pedidos considerados": total_orders,
            "Pedidos planeados": total_orders - postponed_orders,
            "Ordens adiadas": postponed_orders,
            "Caixas adiadas": metrics.get("postponed_boxes", 0),
            "Valor planeado": metrics.get("scheduled_economic_value", 0),
            "Valor adiado": max(0, total_value - metrics.get("scheduled_economic_value", 0)),
            "Utilização de capacidade (%)": compute_overall_capacity_utilization(
                scenario_instance,
                metrics,
            ),
            "Fitness normalizada": metrics.get("normalised_fitness", 0),
        })

    return pd.DataFrame(rows)


def build_postponed_value_df(solution, instance):
    refs_by_id = create_refs_by_id(instance)
    rows = []

    for gene in solution:
        if not gene.get("postponed"):
            continue

        ref_id = str(gene.get("ref_id", "")).strip()
        ref = refs_by_id.get(ref_id, {})
        unit_value = ref.get("economic_value_per_master_box", 0) or 0
        boxes = gene.get("master_boxes", 0) or 0
        rows.append({
            "Referência": ref_id,
            "Classe ABC": get_ref_abc_class(ref),
            "Caixas": boxes,
            "Valor por caixa": round(unit_value, 2),
            "Valor perdido": round(boxes * unit_value, 2),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "Referência",
            "Classe ABC",
            "Caixas",
            "Valor por caixa",
            "Valor perdido",
        ])

    return pd.DataFrame(rows).sort_values("Valor perdido", ascending=False)


def render_automatic_scenario_results(results):
    st.markdown("#### Sensibilidade de capacidade")
    individual_df = results["capacity_individual"]
    cumulative_df = results["capacity_cumulative"]

    if individual_df.empty:
        st.info("Sem dados suficientes para calcular a sensibilidade de capacidade.")
    else:
        heatmap_df = individual_df.pivot_table(
            index="Data",
            columns="Linha mais crítica",
            values="Ganho económico",
            aggfunc="mean",
            fill_value=0,
        )
        heatmap_long = heatmap_df.reset_index().melt(
            id_vars="Data",
            var_name="Linha",
            value_name="Ganho económico",
        )
        fig_heatmap = px.density_heatmap(
            heatmap_long,
            x="Linha",
            y="Data",
            z="Ganho económico",
            color_continuous_scale="Blues",
            title="Impacto económico de adicionar 1 operador",
        )
        st.plotly_chart(fig_heatmap, width="stretch")
        st.dataframe(individual_df, width="stretch", hide_index=True)

    if not cumulative_df.empty:
        fig_cumulative = go.Figure()
        fig_cumulative.add_trace(go.Scatter(
            x=cumulative_df["Operadores adicionais"],
            y=cumulative_df["Ordens adiadas"],
            mode="lines+markers",
            name="Ordens adiadas",
            line={"color": "#b6003b"},
        ))
        fig_cumulative.add_trace(go.Scatter(
            x=cumulative_df["Operadores adicionais"],
            y=cumulative_df["Valor económico produzido"],
            mode="lines+markers",
            name="Valor económico produzido",
            yaxis="y2",
            line={"color": "#153e7e"},
        ))
        fig_cumulative.update_layout(
            title="Impacto acumulado de operadores adicionais",
            xaxis_title="Operadores adicionais",
            yaxis={"title": "Ordens adiadas"},
            yaxis2={
                "title": "Valor económico produzido",
                "overlaying": "y",
                "side": "right",
            },
        )
        st.plotly_chart(fig_cumulative, width="stretch")
        st.dataframe(cumulative_df, width="stretch", hide_index=True)
        recommendation = build_capacity_recommendation(cumulative_df)
        if recommendation:
            st.success(recommendation)

    st.markdown("#### Sensibilidade aos pesos da função objetivo")
    weights_df = results["weights"]
    st.dataframe(
        weights_df.style.apply(style_balanced_profile, axis=1),
        width="stretch",
        hide_index=True,
    )

    if not weights_df.empty:
        radar_metrics = [
            "Ordens adiadas",
            "Caixas adiadas",
            "Valor económico produzido",
            "Setup total",
            "Minutos de operadores usados",
        ]
        fig_radar = go.Figure()

        for _, row in weights_df.iterrows():
            values = []
            for metric in radar_metrics:
                max_value = max(weights_df[metric].max(), 1)
                value = row[metric] / max_value
                if metric in {"Ordens adiadas", "Caixas adiadas", "Setup total"}:
                    value = 1 - value
                values.append(value)

            fig_radar.add_trace(go.Scatterpolar(
                r=values + values[:1],
                theta=radar_metrics + radar_metrics[:1],
                fill="toself",
                name=row["Perfil"],
            ))

        fig_radar.update_layout(
            title="Comparação relativa dos perfis de pesos",
            polar={"radialaxis": {"visible": True, "range": [0, 1]}},
        )
        st.plotly_chart(fig_radar, width="stretch")

    st.markdown("#### Portefólio ABC")
    portfolio_df = results["portfolio"]
    st.dataframe(portfolio_df, width="stretch", hide_index=True)

    if not portfolio_df.empty:
        portfolio_chart_df = pd.concat([
            portfolio_df[[
                "Cenário",
                "Pedidos planeados",
                "Valor planeado",
            ]].rename(columns={
                "Pedidos planeados": "Pedidos",
                "Valor planeado": "Valor económico",
            }).assign(Tipo="Planeados"),
            portfolio_df[[
                "Cenário",
                "Ordens adiadas",
                "Valor adiado",
            ]].rename(columns={
                "Ordens adiadas": "Pedidos",
                "Valor adiado": "Valor económico",
            }).assign(Tipo="Adiados"),
        ], ignore_index=True)
        fig_portfolio = px.bar(
            portfolio_chart_df,
            x="Cenário",
            y="Pedidos",
            color="Tipo",
            text="Valor económico",
            barmode="stack",
            color_discrete_map={
                "Planeados": "#153e7e",
                "Adiados": "#b6003b",
            },
            hover_data=["Valor económico"],
            title="Pedidos planeados vs. adiados por classe ABC",
        )
        fig_portfolio.update_traces(texttemplate="€%{text:,.0f}", textposition="inside")
        st.plotly_chart(fig_portfolio, width="stretch")

    st.markdown("#### Custo dos pedidos adiados no baseline")
    postponed_value_df = results["postponed_value"]
    st.dataframe(postponed_value_df.head(15), width="stretch", hide_index=True)
    if not postponed_value_df.empty:
        n_orders = min(5, len(postponed_value_df))
        top_value = postponed_value_df.head(n_orders)["Valor perdido"].sum()
        st.info(
            f"Os {n_orders} pedidos adiados mais caros representam "
            f"€{top_value:,.0f} de valor não produzido."
        )


def render_automatic_scenario_analysis(instance, baseline_solution, baseline_metrics):
    st.markdown("### Análise automática")
    st.caption(
        "Estas análises correm novas versões do GA com os mesmos parâmetros "
        "calibrados, variando apenas o cenário em estudo."
    )

    if st.button("Executar análise automática", type="primary", width="content"):
        st.session_state["scenario_baseline_solution"] = deepcopy(baseline_solution)
        st.session_state["scenario_baseline_metrics"] = deepcopy(baseline_metrics)
        progress = st.progress(0)
        status = st.empty()

        status.write("A correr 1 de 3: sensibilidade de capacidade...")
        capacity_individual, capacity_cumulative = build_automatic_capacity_results(
            instance,
            baseline_metrics,
        )
        progress.progress(33)

        status.write("A correr 2 de 3: sensibilidade aos pesos...")
        weights_df = build_weight_sensitivity_results(instance)
        progress.progress(66)

        status.write("A correr 3 de 3: análise de portefólio ABC...")
        portfolio_df = build_portfolio_abc_results(instance)
        postponed_value_df = build_postponed_value_df(baseline_solution, instance)
        progress.progress(100)
        status.empty()

        st.session_state["automatic_scenario_analysis"] = {
            "capacity_individual": capacity_individual,
            "capacity_cumulative": capacity_cumulative,
            "weights": weights_df,
            "portfolio": portfolio_df,
            "postponed_value": postponed_value_df,
        }

    results = st.session_state.get("automatic_scenario_analysis")

    if results is None:
        st.info("Clique no botão para gerar os gráficos e tabelas da análise automática.")
        return

    render_automatic_scenario_results(results)


def normalize_dashboard_weights(weights):
    total = sum(max(0, value) for value in weights.values())

    if total <= 0:
        return dict(DEFAULT_NORMALISED_WEIGHTS)

    return {
        key: max(0, value) / total
        for key, value in weights.items()
    }


def render_weight_experiment(instance, baseline_metrics):
    st.markdown("#### Experiência de prioridades")
    st.caption(
        "Ajuste a importância relativa de cada critério. "
        "As prioridades são normalizadas automaticamente para somarem 1."
    )

    weight_labels = {
        "postponement": "Pedidos adiados",
        "economic_value": "Valor económico",
        "delay": "Atraso de entrega",
        "setup": "Tempo de setup",
        "capacity_utilisation": "Utilização de capacidade",
        "operator_utilisation": "Utilização de operadores",
    }
    slider_cols = st.columns(3)
    raw_weights = {}

    for index, (key, label) in enumerate(weight_labels.items()):
        raw_weights[key] = slider_cols[index % 3].slider(
            label,
            min_value=0.0,
            max_value=1.0,
            value=float(DEFAULT_NORMALISED_WEIGHTS.get(key, 0)),
            step=0.01,
            key=f"weight_slider_{key}",
        )

    weights = normalize_dashboard_weights(raw_weights)
    normalized_weight_df = pd.DataFrame([
        {
            "Critério": weight_labels.get(key, key),
            "Prioridade escolhida": round(raw_weights.get(key, 0), 3),
            "Prioridade usada no GA": round(value, 3),
        }
        for key, value in weights.items()
    ])

    st.caption(
        f"Soma das prioridades usadas no GA: {sum(weights.values()):.2f}. "
        "Mesmo que os sliders não somem 1, o algoritmo usa sempre as prioridades normalizadas."
    )
    st.dataframe(
        normalized_weight_df,
        width="stretch",
        hide_index=True,
    )

    if st.button("Simular prioridades", width="content"):
        scenario_instance = deepcopy(instance)
        _, scenario_metrics, _ = run_dashboard_ga_scenario(
            scenario_instance,
            seed=RANDOM_SEED,
            objective_weights=weights,
        )
        st.session_state["weight_experiment_metrics"] = scenario_metrics

    scenario_metrics = st.session_state.get("weight_experiment_metrics")

    if scenario_metrics is not None:
        st.caption(
            "A comparação é feita contra as prioridades equilibradas usadas no plano baseline."
        )
        comparison_df = build_what_if_comparison_df(
            baseline_metrics,
            scenario_metrics,
        )
        st.dataframe(
            comparison_df.style.apply(style_what_if_delta, axis=1),
            width="stretch",
            hide_index=True,
        )
        render_baseline_scenario_bar_chart(
            baseline_metrics,
            scenario_metrics,
            "Prioridades: Equilibrado vs. Cenário",
        )


def build_demand_experiment_df(instance):
    refs_by_id = create_refs_by_id(instance)
    rows = []

    for index, order in enumerate(instance.get("demand", [])):
        ref_id = str(order.get("ref_id", "")).strip()
        rows.append({
            "Ordem": index,
            "Incluir": True,
            "Referência": ref_id,
            "Linha": get_order_line_label(order, refs_by_id),
            "Classe ABC": get_order_abc_class(order, refs_by_id),
            "Caixas": order.get("master_boxes", 0),
            "Entrega": order.get("delivery_date"),
            "Valor económico": round(get_order_economic_value(order, refs_by_id), 2),
        })

    return pd.DataFrame(rows)


def build_signature_status(solution):
    status = {}

    for gene in solution:
        signature = get_solution_signature(gene)
        status.setdefault(signature, set()).add(
            "Adiado" if gene.get("postponed") else "Planeado"
        )

    return status


def render_demand_experiment(instance, baseline_solution, baseline_metrics):
    st.markdown("#### What-If de procura")
    st.caption(
        "Varie a procura do cenário ao incluir/excluir pedidos ou alterar quantidades. "
        "O GA volta a correr apenas com a procura definida nesta tabela."
    )

    demand_df = build_demand_experiment_df(instance)
    edited_df = st.data_editor(
        demand_df,
        column_config={
            "Incluir": st.column_config.CheckboxColumn(),
            "Caixas": st.column_config.NumberColumn(min_value=0, step=1, format="%d"),
        },
        disabled=["Ordem", "Referência", "Linha", "Classe ABC", "Entrega", "Valor económico"],
        hide_index=True,
        width="stretch",
        height=360,
        key="demand_experiment_editor",
    )

    if st.button("Simular procura", width="content"):
        selected_demand = []

        for _, row in edited_df[edited_df["Incluir"]].iterrows():
            order_index = int(row["Ordem"])
            order = deepcopy(instance.get("demand", [])[order_index])
            order["master_boxes"] = int(max(0, row.get("Caixas", 0) or 0))
            selected_demand.append(order)

        scenario_instance = make_scenario_instance(instance, demand=selected_demand)
        scenario_solution, scenario_metrics, _ = run_dashboard_ga_scenario(
            scenario_instance,
            seed=RANDOM_SEED,
        )
        st.session_state["demand_experiment_instance"] = scenario_instance
        st.session_state["demand_experiment_solution"] = scenario_solution
        st.session_state["demand_experiment_metrics"] = scenario_metrics

    scenario_metrics = st.session_state.get("demand_experiment_metrics")
    scenario_solution = st.session_state.get("demand_experiment_solution")
    scenario_instance = st.session_state.get("demand_experiment_instance")

    if scenario_metrics is None or scenario_solution is None or scenario_instance is None:
        return

    total_orders = len(scenario_instance.get("demand", []))
    postponed_orders = scenario_metrics.get("postponed_orders", 0)
    fulfilment_rate = (
        (total_orders - postponed_orders) / total_orders * 100
        if total_orders
        else 0
    )
    capacity_utilization = compute_overall_capacity_utilization(
        scenario_instance,
        scenario_metrics,
    )
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    kpi_col1.metric("Taxa de cumprimento", f"{fulfilment_rate:.1f}%")
    kpi_col2.metric(
        "Valor produzido",
        f"€{scenario_metrics.get('scheduled_economic_value', 0):,.0f}",
    )
    kpi_col3.metric("Utilização de capacidade", f"{capacity_utilization:.1f}%")

    comparison_df = build_what_if_comparison_df(
        baseline_metrics,
        scenario_metrics,
    )
    st.dataframe(
        comparison_df.style.apply(style_what_if_delta, axis=1),
        width="stretch",
        hide_index=True,
    )
    render_baseline_scenario_bar_chart(
        baseline_metrics,
        scenario_metrics,
        "Procura: Baseline vs. Cenário",
    )

    baseline_status = build_signature_status(baseline_solution)
    scenario_status = build_signature_status(scenario_solution)
    movement_rows = []

    for signature, statuses in baseline_status.items():
        scenario_statuses = scenario_status.get(signature, set())
        if "Adiado" in statuses and "Planeado" in scenario_statuses:
            movement = "Recuperado"
        elif "Planeado" in statuses and "Adiado" in scenario_statuses:
            movement = "Novo adiamento"
        else:
            continue

        ref_id, boxes, delivery = signature
        movement_rows.append({
            "Referência": ref_id,
            "Caixas": boxes,
            "Entrega": delivery,
            "Movimento": movement,
        })

    if movement_rows:
        st.markdown("**Pedidos que mudaram de estado**")
        st.dataframe(pd.DataFrame(movement_rows), width="stretch", hide_index=True)


def render_experimental_scenario_analysis(instance, baseline_solution, baseline_metrics):
    render_capacity_what_if_section(instance, baseline_solution, baseline_metrics)
    st.divider()
    render_weight_experiment(instance, baseline_metrics)
    st.divider()
    render_demand_experiment(instance, baseline_solution, baseline_metrics)


def render_scenario_analysis_module(instance, baseline_solution, baseline_metrics):
    st.subheader("What-If Analysis")
    st.caption(
        "Use esta área para testar cenários operacionais simples, sem alterar "
        "o plano baseline."
    )
    render_experimental_scenario_analysis(
        instance,
        baseline_solution,
        baseline_metrics,
    )


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
                f"{row['Reference']} ({row['Master boxes']} caixas)"
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
            row[f"{line} preparação"] = f"{setup_time:.1f} min"
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
    breakdown = normalised_fitness_breakdown(metrics, max_values)
    score = normalised_fitness(metrics, max_values)

    metrics["normalised_fitness"] = score
    metrics["normalised_fitness_breakdown"] = breakdown
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
                qty_col = f"{ref} | caixas"
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
    exact_intervals = best_metrics.get("operator_usage_intervals", [])
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
            overlapping_intervals = [
                interval for interval in exact_intervals
                if (
                    interval["day"] == day
                    and interval["start"] < end
                    and interval["end"] > start
                )
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

            if overlapping_intervals:
                row["Total operators used"] = max(
                    interval.get("operators", 0)
                    for interval in overlapping_intervals
                )
                row["Operator excess"] = max(
                    interval.get("excess", 0)
                    for interval in overlapping_intervals
                )
            else:
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


def build_contribution_df(best_metrics):
    labels = {
        "postponement": "Adiamento",
        "delay": "Atraso",
        "setup": "Preparação",
        "economic_value": "Valor económico",
        "capacity_utilisation": "Utilização de capacidade",
        "operator_utilisation": "Utilização de operadores",
    }
    rows = [
        {
            "Componente": labels.get(component, component),
            "Contribuição normalizada": value,
        }
        for component, value in best_metrics.get(
            "normalised_fitness_breakdown",
            {},
        ).items()
        if component != "total"
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


def get_ga_context():
    required = ["ga_solution", "ga_metrics", "ga_instance"]

    if not all(key in st.session_state for key in required):
        return None

    return (
        deepcopy(st.session_state["ga_instance"]),
        deepcopy(st.session_state["ga_solution"]),
        deepcopy(st.session_state["ga_metrics"]),
    )


def scenario_ga_params():
    return {
        "population_size": POPULATION_SIZE,
        "generations": GENERATIONS,
        "mutation_rate": MUTATION_RATE,
        "elite_size": ELITE_SIZE,
        "tournament_size": TOURNAMENT_SIZE,
        "stagnation_k": STAGNATION_K,
        "seed": RANDOM_SEED,
    }


def render_scenario_result(name, result_df):
    if result_df is None or result_df.empty:
        return

    if name == "Capacidade":
        chart = (
            alt.Chart(result_df)
            .mark_line(point=True, color="#153e7e", strokeWidth=3)
            .encode(
                x=alt.X("Capacidade (%):Q", title="Capacidade disponível (%)"),
                y=alt.Y("Ordens adiadas:Q", title="Ordens adiadas"),
                tooltip=list(result_df.columns),
            )
            .properties(
                title="Ordens adiadas em função da capacidade",
                height=360,
            )
        )
    elif name == "Pesos":
        chart = (
            alt.Chart(result_df)
            .mark_circle(size=130)
            .encode(
                x=alt.X("Ordens adiadas:Q", title="Ordens adiadas"),
                y=alt.Y(
                    "Valor económico planeado:Q",
                    title="Valor económico planeado (€)",
                ),
                color=alt.Color(
                    "Peso adiamento:Q",
                    title="Peso do adiamento",
                    scale=alt.Scale(range=["#9db4d5", "#153e7e"]),
                ),
                tooltip=list(result_df.columns),
            )
            .properties(
                title="Compromisso entre adiamento e valor económico",
                height=360,
            )
        )
    else:
        chart = (
            alt.Chart(result_df)
            .mark_line(point=True, color="#b6003b", strokeWidth=3)
            .encode(
                x=alt.X(
                    "Multiplicador da procura:Q",
                    title="Multiplicador da procura",
                ),
                y=alt.Y("Ordens adiadas:Q", title="Ordens adiadas"),
                tooltip=list(result_df.columns),
            )
            .properties(
                title="Degradação do plano com o aumento da procura",
                height=360,
            )
        )

    st.altair_chart(chart, width="stretch")
    render_interactive_table(
        result_df,
        key=f"analise_{name.lower()}",
        height=360,
    )


def render_scenario_analysis():
    st.header("Análise de cenários")
    context = get_ga_context()

    if context is None:
        st.info(
            "Gere primeiro o plano no separador Configuração e Plano. "
            "As análises usam exatamente a mesma instância e parâmetros."
        )
        return

    instance, _, _ = context
    analysis_labels = {
        "Sensibilidade da capacidade": "Capacidade",
        "Sensibilidade dos pesos": "Pesos",
        "Teste de esforço do volume da procura": "Procura",
    }
    selected_label = st.radio(
        "Análise a visualizar",
        options=list(analysis_labels),
        horizontal=True,
    )
    selected_name = analysis_labels[selected_label]
    button_col1, button_col2 = st.columns([1, 1])
    run_selected = button_col1.button(
        "Executar análise selecionada",
        type="primary",
        width="stretch",
    )
    run_all = button_col2.button(
        "Executar todas as análises",
        width="stretch",
    )
    st.caption(
        "Cada ponto usa a semente aleatória 42 e os mesmos parâmetros do plano principal. "
        "Os resultados ficam guardados durante a sessão."
    )

    if "scenario_results" not in st.session_state:
        st.session_state["scenario_results"] = {}

    analyses = {
        "Capacidade": capacity_sensitivity,
        "Pesos": weight_sensitivity,
        "Procura": demand_stress_test,
    }

    if run_selected or run_all:
        names_to_run = list(analyses) if run_all else [selected_name]
        progress_bar = st.progress(0)
        progress_text = st.empty()

        for analysis_index, name in enumerate(names_to_run):
            progress_text.markdown(f"**A calcular: {name}**")

            def update_progress(done, total, message, index=analysis_index):
                fraction = (index + done / max(1, total)) / len(names_to_run)
                progress_bar.progress(min(1.0, fraction))
                progress_text.markdown(f"**{name}: {message}**")

            result = analyses[name](
                deepcopy(instance),
                scenario_ga_params(),
                progress=update_progress,
            )
            st.session_state["scenario_results"][name] = result
            progress_bar.progress(
                (analysis_index + 1) / len(names_to_run)
            )

        progress_text.markdown("**Análises concluídas.**")

    result_df = st.session_state["scenario_results"].get(selected_name)

    if result_df is None:
        st.info("Esta análise ainda não foi executada.")
    else:
        render_scenario_result(selected_name, result_df)


def performance_comparison_df(ga_metrics, baseline_metrics):
    rows = [
        ("Aptidão normalizada", "normalised_fitness"),
        ("Ordens adiadas", "postponed_orders"),
        ("Caixas adiadas", "postponed_boxes"),
        ("Atraso total (dias)", "delay_days_total"),
        ("Preparação total (min)", "setup_total_min"),
        ("Valor económico planeado", "scheduled_economic_value"),
        ("Utilização de operadores (min)", "operator_usage_minutes"),
    ]
    return pd.DataFrame([
        {
            "Indicador": label,
            "Plano GA": ga_metrics.get(key, 0),
            "Referência sem otimização": baseline_metrics.get(key, 0),
        }
        for label, key in rows
    ])


def build_line_metrics_summary_df(instance, solution, metrics):
    plan_df = build_plan_df(instance, solution)
    scheduled_df = plan_df[plan_df["Status"] == "Scheduled"].copy()
    rows = []

    for line in instance.get("final_lines", ["L1", "L2"]):
        line_df = scheduled_df[scheduled_df["Line"] == line]
        total_minutes = line_df["Production time (min)"].sum()
        total_euros = line_df["Economic value"].sum()
        total_kg = line_df["Kg"].sum()

        rows.append({
            "Grupo": line,
            "Tempo total (h)": round(total_minutes / 60, 2),
            "Tempo total (min)": round(total_minutes, 1),
            "Euros totais": round(total_euros, 2),
            "Kg totais": round(total_kg, 2),
        })

    total_minutes = scheduled_df["Production time (min)"].sum()

    rows.append({
        "Grupo": "Total planeado",
        "Tempo total (h)": round(total_minutes / 60, 2),
        "Tempo total (min)": round(total_minutes, 1),
        "Euros totais": round(metrics.get("scheduled_economic_value", 0), 2),
        "Kg totais": round(metrics.get("scheduled_kg", 0), 2),
    })
    rows.append({
        "Grupo": "Adiado",
        "Tempo total (h)": "",
        "Tempo total (min)": "",
        "Euros totais": round(metrics.get("postponed_economic_value", 0), 2),
        "Kg totais": round(metrics.get("postponed_kg", 0), 2),
    })

    return pd.DataFrame(rows)


def render_performance_metrics():
    st.header("Métricas de desempenho")
    context = get_ga_context()

    if context is None:
        st.info("Gere primeiro o plano no separador Configuração e Plano.")
        return

    instance, solution, metrics = context
    line_summary_df = build_line_metrics_summary_df(
        instance,
        solution,
        metrics,
    )

    st.subheader("Resumo por linha")
    st.dataframe(
        line_summary_df.style.format({
            "Tempo total (h)": "{}",
            "Tempo total (min)": "{}",
            "Euros totais": "€ {:,.2f}",
            "Kg totais": "{:,.2f}",
        }),
        width="stretch",
        hide_index=True,
        height=220,
    )

    breakdown = metrics.get("normalised_fitness_breakdown", {})
    breakdown_df = pd.DataFrame([
        {
            "Componente": component,
            "Contribuição": value,
        }
        for component, value in breakdown.items()
        if component != "total"
    ])
    labels = {
        "postponement": "Adiamento",
        "delay": "Atraso",
        "setup": "Preparação",
        "economic_value": "Valor económico",
        "operator_utilisation": "Utilização de operadores",
    }
    breakdown_df["Componente"] = breakdown_df["Componente"].replace(labels)

    breakdown_chart = (
        alt.Chart(breakdown_df)
        .mark_bar()
        .encode(
            x=alt.X("Contribuição:Q", title="Contribuição para a aptidão"),
            y=alt.Y(
                "Componente:N",
                title=None,
                sort="-x",
            ),
            color=alt.condition(
                alt.datum["Contribuição"] < 0,
                alt.value("#153e7e"),
                alt.value("#b6003b"),
            ),
            tooltip=["Componente", alt.Tooltip("Contribuição:Q", format=".6f")],
        )
        .properties(
            title="Decomposição da aptidão normalizada",
            height=270,
        )
    )
    st.altair_chart(breakdown_chart, width="stretch")

    capacity_df = build_capacity_df(instance, metrics)

    if not capacity_df.empty:
        capacity_chart_df = capacity_df.rename(columns={
            "Production date": "Data de produção",
            "Day": "Dia",
            "Line": "Linha",
            "Shifts": "Turnos",
            "Production time (min)": "Tempo de produção (min)",
            "Setup time (min)": "Tempo de preparação (min)",
            "Occupied time (min)": "Tempo ocupado (min)",
            "Available time (min)": "Tempo disponível (min)",
            "Capacity excess (min)": "Excesso de capacidade (min)",
            "Utilization (%)": "Utilização (%)",
        })
        capacity_chart = (
            alt.Chart(capacity_chart_df)
            .mark_bar()
            .encode(
                x=alt.X("Data de produção:N", title="Data"),
                xOffset="Linha:N",
                y=alt.Y("Utilização (%):Q", title="Utilização (%)"),
                color=alt.Color(
                    "Linha:N",
                    title="Linha",
                    scale=alt.Scale(range=LINE_COLORS),
                ),
                tooltip=list(capacity_chart_df.columns),
            )
            .properties(
                title="Utilização diária da capacidade por linha",
                height=340,
            )
        )
        st.altair_chart(capacity_chart, width="stretch")

    signature = st.session_state.get("ga_instance_signature")

    if st.session_state.get("baseline_signature") != signature:
        with st.spinner("A construir a referência sem otimização..."):
            baseline_solution, baseline_metrics = build_greedy_baseline(
                deepcopy(instance)
            )
        st.session_state["baseline_solution"] = baseline_solution
        st.session_state["baseline_metrics"] = baseline_metrics
        st.session_state["baseline_signature"] = signature

    comparison_df = performance_comparison_df(
        metrics,
        st.session_state["baseline_metrics"],
    )
    st.subheader("GA vs. referência sem otimização")
    render_interactive_table(
        comparison_df,
        key="comparacao_ga_baseline",
        height=330,
    )


def metrics_summary_df(metrics):
    keys = [
        ("Aptidão normalizada", "normalised_fitness"),
        ("Ordens adiadas", "postponed_orders"),
        ("Caixas adiadas", "postponed_boxes"),
        ("Atraso total (dias)", "delay_days_total"),
        ("Preparação total (min)", "setup_total_min"),
        ("Valor económico planeado", "scheduled_economic_value"),
        ("Valor económico adiado", "postponed_economic_value"),
        ("Kg planeados", "scheduled_kg"),
        ("Kg adiados", "postponed_kg"),
        ("Operadores-minuto utilizados", "operator_usage_minutes"),
        ("Tempo computacional (s)", "computation_time_sec"),
    ]
    return pd.DataFrame([
        {"Indicador": label, "Valor": metrics.get(key, 0)}
        for label, key in keys
    ])


def render_export():
    st.header("Exportação")
    context = get_ga_context()

    if context is None:
        st.info("Gere primeiro o plano no separador Configuração e Plano.")
        return

    instance, solution, metrics = context
    plan_df = build_plan_df(instance, solution)
    scenario_frames = list(
        st.session_state.get("scenario_results", {}).values()
    )
    scenario_df = (
        pd.concat(scenario_frames, ignore_index=True, sort=False)
        if scenario_frames
        else pd.DataFrame()
    )
    summary_df = metrics_summary_df(metrics)
    col1, col2, col3 = st.columns(3)

    col1.download_button(
        "Descarregar plano CSV",
        plan_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="plano_producao.csv",
        mime="text/csv",
        width="stretch",
    )
    col2.download_button(
        "Descarregar cenários CSV",
        scenario_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="resultados_cenarios.csv",
        mime="text/csv",
        disabled=scenario_df.empty,
        width="stretch",
    )
    col3.download_button(
        "Descarregar métricas CSV",
        summary_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="resumo_metricas.csv",
        mime="text/csv",
        width="stretch",
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


def render_configuration_plan():
    st.subheader("Ficheiro de dados de entrada")
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
        st.error(f"Não foi possível carregar o ficheiro de dados de entrada: {exc}")
        return
    
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
        return
    
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
        return
    
    default_operators = DEFAULT_OPERATORS
    operators_by_day = {}
    start_times_by_day = {}
    end_times_by_day = {}
    shifts_by_day = {}
    
    with st.expander("Operadores e horários por dia", expanded=True):
        st.caption(
            "Os valores abaixo usam os valores predefinidos como ponto de partida, "
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
                    stagnation_k=STAGNATION_K,
                    seed=RANDOM_SEED,
                )
    
            st.session_state["ga_solution"] = deepcopy(best_solution)
            st.session_state["ga_metrics"] = deepcopy(best_metrics)
            st.session_state["ga_instance"] = deepcopy(instance)
            st.session_state["ga_planning_month"] = planning_month
            st.session_state["ga_instance_signature"] = current_instance_signature
            st.session_state["scenario_results"] = {}
            st.session_state.pop("capacity_what_if_instance", None)
            st.session_state.pop("capacity_what_if_solution", None)
            st.session_state.pop("capacity_what_if_metrics", None)
            st.session_state.pop("capacity_what_if_input", None)
            st.session_state.pop("baseline_signature", None)
            st.session_state["scenario_version"] = (
                st.session_state.get("scenario_version", 0) + 1
            )
        except Exception as exc:
            st.error(f"Não foi possível gerar o plano: {exc}")
            return
    
    if "ga_solution" not in st.session_state:
        st.info("Pressione o botão para gerar o plano de produção.")
        return
    
    if st.session_state.get("ga_instance_signature") != current_instance_signature:
        for session_key in [
            "ga_solution",
            "ga_metrics",
            "ga_instance",
            "ga_planning_month",
            "ga_instance_signature",
        ]:
            st.session_state.pop(session_key, None)
    
        st.warning(
            "Os dados de entrada ou os parâmetros operacionais foram alterados. "
            "Gere novamente o plano para evitar misturar uma solução antiga "
            "com a instância atual."
        )
        return
    
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
        return
    
    max_values = compute_max_values(instance)
    best_metrics = add_normalised_fitness_metrics(
        evaluate_solution(best_solution, instance),
        max_values,
    )
    st.session_state["ga_metrics"] = deepcopy(best_metrics)
    st.session_state["ga_instance"] = deepcopy(instance)
    planning_month = st.session_state.get(
        "ga_planning_month",
        get_planning_month(instance),
    )
    
    st.success(f"Plano gerado com sucesso para {planning_month}.")
    
    plan_df = build_plan_df(instance, best_solution)
    simple_plan_df = build_simple_daily_plan_df(instance, plan_df)
    daily_product_schedule_df = build_daily_product_schedule_df(
        instance,
        plan_df,
        best_metrics,
    )
    capacity_df = build_capacity_df(instance, best_metrics)
    time_slot_df = build_time_slot_activity_df(instance, best_metrics)
    
    st.subheader("Indicadores principais")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Kg produzidos", f"{best_metrics.get('scheduled_kg', 0):,.1f}")
    col2.metric("Kg adiados", f"{best_metrics.get('postponed_kg', 0):,.1f}")
    col3.metric("Valor produzido", f"€{best_metrics.get('scheduled_economic_value', 0):,.0f}")
    col4.metric("Valor adiado", f"€{best_metrics.get('postponed_economic_value', 0):,.0f}")

    render_scenario_analysis_module(
        instance,
        best_solution,
        best_metrics,
    )
    
    st.subheader("Plano diário simplificado")
    render_simple_daily_plan_table(
        simple_plan_df,
        key="plano_diario_simplificado",
        height=620,
    )

    st.subheader("Sequência diária de produção")
    render_interactive_table(
        daily_product_schedule_df,
        key="sequencia_diaria_producao",
        height=420,
    )
    
    st.subheader("Ocupação de operadores por horário")
    if time_slot_df.empty:
        st.info("Não existe simulação horária disponível.")
    else:
        operators_chart_df = time_slot_df.rename(columns={
            "Production date": "Data de produção",
            "Day": "Dia",
            "Time slot": "Horário",
            "Slot start (min)": "Início do intervalo (min)",
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
        hourly_chart = (
            alt.Chart(operators_chart_df)
            .mark_rect()
            .encode(
                x=alt.X(
                    "Horário:N",
                    title="Horário",
                    sort=alt.SortField(
                        field="Início do intervalo (min)",
                        order="ascending",
                    ),
                ),
                y=alt.Y("Data de produção:N", title="Data de produção"),
                color=alt.Color(
                    "Operadores usados:Q",
                    title="Operadores usados",
                    scale=alt.Scale(
                        domain=[0, max(1, time_slot_df["Total operators used"].max())],
                        range=["#f1f5f9", "#153e7e", "#b6003b"],
                    ),
                ),
                tooltip=list(operators_chart_df.columns),
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
                    "setup": "preparação",
                    "production | setup": "produção | preparação",
                    "finishing | production": "acabamento | produção",
                    "finishing | setup": "acabamento | preparação",
                    "finishing | production | setup": "acabamento | produção | preparação",
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
        capacity_chart_df = capacity_df.rename(columns={
            "Production date": "Data de produção",
            "Day": "Dia",
            "Line": "Linha",
            "Shifts": "Turnos",
            "Production time (min)": "Tempo de produção (min)",
            "Setup time (min)": "Tempo de preparação (min)",
            "Occupied time (min)": "Tempo ocupado (min)",
            "Available time (min)": "Tempo disponível (min)",
            "Capacity excess (min)": "Excesso de capacidade (min)",
            "Utilization (%)": "Utilização (%)",
        })
        capacity_chart = (
            alt.Chart(capacity_chart_df)
            .mark_bar()
            .encode(
                x=alt.X("Data de produção:N", title="Data de produção"),
                xOffset=alt.XOffset("Linha:N"),
                y=alt.Y("Utilização (%):Q", title="Utilização (%)"),
                color=alt.Color("Linha:N", title="Linha", scale=alt.Scale(range=LINE_COLORS)),
                tooltip=list(capacity_chart_df.columns),
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
            "Setup time (min)": "Tempo de preparação (min)",
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
    
render_configuration_plan()
