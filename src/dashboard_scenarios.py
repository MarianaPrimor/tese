from copy import deepcopy
import time

import pandas as pd

from evaluator import (
    compute_max_values,
    create_refs_by_id,
    evaluate_solution,
    get_valid_days_for_ref,
    normalised_fitness_breakdown,
    valid_lines_for_ref,
)
from geneticalgorithm import run_genetic_algorithm


CAPACITY_LEVELS = [70, 80, 90, 100, 110, 120]
POSTPONEMENT_WEIGHTS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
DEMAND_MULTIPLIERS = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

DEFAULT_WEIGHTS = {
    "postponement": 0.40,
    "economic_value": 0.30,
    "delay": 0.15,
    "setup": 0.10,
    "operator_utilisation": 0.05,
}


def add_normalised_metrics(metrics, instance, weights=None):
    max_values = compute_max_values(instance)
    breakdown = normalised_fitness_breakdown(
        metrics,
        max_values,
        weights=weights,
    )
    metrics["normalised_fitness_breakdown"] = breakdown
    metrics["normalised_fitness"] = breakdown["total"]
    metrics["max_values"] = max_values
    return metrics


def run_ga_scenario(
    instance,
    ga_params,
    objective_weights=None,
):
    start_time = time.perf_counter()
    solution, metrics, generations = run_genetic_algorithm(
        instance,
        population_size=ga_params["population_size"],
        generations=ga_params["generations"],
        mutation_rate=ga_params["mutation_rate"],
        elite_size=ga_params["elite_size"],
        tournament_size=ga_params["tournament_size"],
        stagnation_k=ga_params["stagnation_k"],
        seed=ga_params["seed"],
        verbose=False,
        objective_weights=objective_weights,
    )
    metrics["scenario_computation_time_sec"] = (
        time.perf_counter() - start_time
    )
    metrics["scenario_generations"] = generations
    return solution, metrics


def capacity_instance(instance, percentage):
    scenario = deepcopy(instance)
    factor = percentage / 100
    base_capacity = scenario.get("daily_capacity_min", {})

    if base_capacity:
        scenario["daily_capacity_min"] = {
            day: capacity * factor
            for day, capacity in base_capacity.items()
        }
        scenario["available_line_time_min"] = min(
            scenario["daily_capacity_min"].values()
        )
    else:
        scenario["available_line_time_min"] *= factor

    return scenario


def demand_instance(instance, multiplier):
    scenario = deepcopy(instance)

    for order in scenario.get("demand", []):
        order["master_boxes"] = max(
            1,
            int(round(order.get("master_boxes", 0) * multiplier)),
        )

    return scenario


def capacity_sensitivity(instance, ga_params, progress=None):
    rows = []

    for index, percentage in enumerate(CAPACITY_LEVELS, start=1):
        if progress:
            progress(index - 1, len(CAPACITY_LEVELS), f"Capacidade {percentage}%")

        scenario = capacity_instance(instance, percentage)
        _, metrics = run_ga_scenario(scenario, ga_params)
        rows.append({
            "Análise": "Sensibilidade da capacidade",
            "Capacidade (%)": percentage,
            "Ordens adiadas": metrics["postponed_orders"],
            "Caixas adiadas": metrics["postponed_boxes"],
            "Valor económico planeado": metrics["scheduled_economic_value"],
            "Fitness normalizada": metrics["normalised_fitness"],
            "Tempo de setup (min)": metrics["setup_total_min"],
            "Tempo computacional (s)": metrics["scenario_computation_time_sec"],
        })

    if progress:
        progress(len(CAPACITY_LEVELS), len(CAPACITY_LEVELS), "Concluída")

    return pd.DataFrame(rows)


def weight_sensitivity(instance, ga_params, progress=None):
    rows = []
    fixed_weight = (
        DEFAULT_WEIGHTS["delay"]
        + DEFAULT_WEIGHTS["setup"]
        + DEFAULT_WEIGHTS["operator_utilisation"]
    )
    available_pair_weight = 1 - fixed_weight

    for index, postponement_weight in enumerate(
        POSTPONEMENT_WEIGHTS,
        start=1,
    ):
        economic_weight = available_pair_weight - postponement_weight
        weights = {
            **DEFAULT_WEIGHTS,
            "postponement": postponement_weight,
            "economic_value": economic_weight,
        }

        if progress:
            progress(
                index - 1,
                len(POSTPONEMENT_WEIGHTS),
                (
                    f"Adiamento {postponement_weight:.2f} / "
                    f"valor {economic_weight:.2f}"
                ),
            )

        _, metrics = run_ga_scenario(
            deepcopy(instance),
            ga_params,
            objective_weights=weights,
        )
        rows.append({
            "Análise": "Sensibilidade dos pesos",
            "Peso adiamento": postponement_weight,
            "Peso valor económico": economic_weight,
            "Ordens adiadas": metrics["postponed_orders"],
            "Caixas adiadas": metrics["postponed_boxes"],
            "Valor económico planeado": metrics["scheduled_economic_value"],
            "Fitness normalizada": metrics["normalised_fitness"],
            "Atraso total (dias)": metrics["delay_days_total"],
            "Tempo de setup (min)": metrics["setup_total_min"],
            "Tempo computacional (s)": metrics["scenario_computation_time_sec"],
        })

    if progress:
        progress(
            len(POSTPONEMENT_WEIGHTS),
            len(POSTPONEMENT_WEIGHTS),
            "Concluída",
        )

    return pd.DataFrame(rows)


def demand_stress_test(instance, ga_params, progress=None):
    rows = []

    for index, multiplier in enumerate(DEMAND_MULTIPLIERS, start=1):
        if progress:
            progress(
                index - 1,
                len(DEMAND_MULTIPLIERS),
                f"Procura {multiplier:.1f}x",
            )

        scenario = demand_instance(instance, multiplier)
        _, metrics = run_ga_scenario(scenario, ga_params)
        rows.append({
            "Análise": "Stress da procura",
            "Multiplicador da procura": multiplier,
            "Procura total (caixas)": sum(
                order.get("master_boxes", 0)
                for order in scenario["demand"]
            ),
            "Ordens adiadas": metrics["postponed_orders"],
            "Caixas adiadas": metrics["postponed_boxes"],
            "Valor económico planeado": metrics["scheduled_economic_value"],
            "Fitness normalizada": metrics["normalised_fitness"],
            "Tempo de setup (min)": metrics["setup_total_min"],
            "Tempo computacional (s)": metrics["scenario_computation_time_sec"],
        })

    if progress:
        progress(
            len(DEMAND_MULTIPLIERS),
            len(DEMAND_MULTIPLIERS),
            "Concluída",
        )

    return pd.DataFrame(rows)


def build_greedy_baseline(instance):
    refs_by_id = create_refs_by_id(instance)
    ordered_ids = sorted(
        range(len(instance["demand"])),
        key=lambda order_id: (
            instance["demand"][order_id].get("delivery_date")
            if instance["demand"][order_id].get("delivery_date") is not None
            else instance["n_days"] + 1,
            order_id,
        ),
    )
    scheduled = []
    postponed = {}

    for order_id in ordered_ids:
        order = instance["demand"][order_id]
        ref_id = str(order["ref_id"]).strip()
        ref = refs_by_id.get(ref_id)
        base_gene = {
            "order_id": order_id,
            "ref_id": ref_id,
            "master_boxes": order["master_boxes"],
            "delivery_date": order.get("delivery_date"),
            "delivery_calendar_date": order.get("delivery_calendar_date"),
            "adjusted_delivery_date": order.get("adjusted_delivery_date"),
            "priority": order.get("priority", "Medium"),
        }

        if ref is None:
            postponed[order_id] = {
                **base_gene,
                "day": None,
                "line": None,
                "postponed": True,
            }
            continue

        assigned = False
        valid_days = sorted(
            get_valid_days_for_ref(instance, ref),
            key=lambda day: (
                day > (order.get("delivery_date") or instance["n_days"]),
                day,
            ),
        )

        for day in valid_days:
            for line in valid_lines_for_ref(ref):
                candidate_gene = {
                    **base_gene,
                    "day": day,
                    "line": line,
                    "postponed": False,
                }
                provisional = scheduled + [candidate_gene]
                provisional.extend(
                    {
                        "order_id": remaining_id,
                        "ref_id": str(
                            instance["demand"][remaining_id]["ref_id"]
                        ).strip(),
                        "master_boxes": instance["demand"][remaining_id][
                            "master_boxes"
                        ],
                        "delivery_date": instance["demand"][remaining_id].get(
                            "delivery_date"
                        ),
                        "day": None,
                        "line": None,
                        "postponed": True,
                    }
                    for remaining_id in ordered_ids
                    if remaining_id not in {
                        gene["order_id"] for gene in provisional
                    }
                )
                metrics = evaluate_solution(provisional, instance)

                if not metrics["infeasible_solution"]:
                    scheduled.append(candidate_gene)
                    assigned = True
                    break

            if assigned:
                break

        if not assigned:
            postponed[order_id] = {
                **base_gene,
                "day": None,
                "line": None,
                "postponed": True,
            }

    solution = scheduled + [
        postponed[order_id]
        for order_id in ordered_ids
        if order_id in postponed
    ]
    metrics = add_normalised_metrics(
        evaluate_solution(solution, instance),
        instance,
    )
    return solution, metrics
