import gurobipy as gp
from gurobipy import GRB
import time

from generate_instance import load_real_instance, calculate_production_time
from evaluator import (
    create_refs_by_id,
    valid_lines_for_ref,
    generate_random_solution,
    evaluate_solution,
    print_metrics,
    print_validation_report,
    get_valid_days_for_ref,
    get_production_time as evaluator_get_production_time,
    get_setup,
    get_production_operators,
    get_finishing_operators,
    get_available_line_time_for_day,
    get_capacity_tolerance_for_day,
    compute_max_values,
    normalised_fitness,
    normalised_fitness_breakdown,
    POSTPONEMENT_NORMALIZED_WEIGHT,
    ECONOMIC_VALUE_NORMALIZED_WEIGHT,
    DELAY_NORMALIZED_WEIGHT,
    SETUP_NORMALIZED_WEIGHT,
    CAPACITY_UTILIZATION_NORMALIZED_WEIGHT,
    OPERATOR_UTILIZATION_NORMALIZED_WEIGHT,
    FINISHING_DELAY_L1_MIN,
)
from geneticalgorithm import enforce_hard_constraints


def get_production_time(ref, line, master_boxes):
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

def get_required_operators(ref, line):
    if line == "L1":
        return (ref["ops_L1_prod"] or 0) + (ref["ops_L1_finish"] or 0)

    if line == "L2":
        return (ref["ops_L2_prod"] or 0) + (ref["ops_L2_finish"] or 0)

    return 0

def get_selected_assignment(x, order, lines, days):
    for line in lines:
        for day in days:
            if x[order, line, day].X > 0.5:
                return line, day

    return None, None


def get_calendar_date_for_day(instance, day):
    working_days = instance.get("working_days", [])

    if day is not None and 1 <= day <= len(working_days):
        return working_days[day - 1]

    return None



def _selected_assignment(x, order, lines, days):
    for line in lines:
        for day in days:
            if x[order, line, day].X > 0.5:
                return line, day

    return None, None


def extract_full_solution(instance, x, postponed, orders, lines, days, positions):
    solution = []
    selected_variables = []

    for l in lines:
        for d in days:
            for p in positions:
                for o in orders:
                    if x[o, l, d, p].X <= 0.5:
                        continue

                    order = instance["demand"][o]
                    selected_variables.append(x[o, l, d, p])
                    solution.append({
                        "order_id": o,
                        "ref_id": str(order["ref_id"]).strip(),
                        "master_boxes": order["master_boxes"],
                        "delivery_date": order["delivery_date"],
                        "delivery_calendar_date": order.get("delivery_calendar_date"),
                        "adjusted_delivery_date": order.get("adjusted_delivery_date"),
                        "priority": order.get("priority", "Medium"),
                        "day": d,
                        "line": l,
                        "postponed": False,
                    })

    for o in orders:
        if postponed[o].X <= 0.5:
            continue

        order = instance["demand"][o]
        selected_variables.append(postponed[o])
        solution.append({
            "order_id": o,
            "ref_id": str(order["ref_id"]).strip(),
            "master_boxes": order["master_boxes"],
            "delivery_date": order["delivery_date"],
            "delivery_calendar_date": order.get("delivery_calendar_date"),
            "adjusted_delivery_date": order.get("adjusted_delivery_date"),
            "priority": order.get("priority", "Medium"),
            "day": None,
            "line": None,
            "postponed": True,
        })

    return solution, selected_variables


def apply_feasible_mip_start(
    model,
    instance,
    x,
    postponed,
    orders,
    lines,
    days,
    positions,
    seed=42,
):
    """Seed Gurobi with a feasible plan built by the GA hard-constraint repair."""
    initial_solution = enforce_hard_constraints(
        generate_random_solution(instance, seed=seed),
        instance,
    )
    genes_by_order = {
        gene["order_id"]: gene
        for gene in initial_solution
    }
    next_position = {
        (line, day): 1
        for line in lines
        for day in days
    }

    for o in orders:
        gene = genes_by_order.get(o)

        if gene is None or gene.get("postponed"):
            postponed[o].Start = 1
            continue

        line = gene.get("line")
        day = gene.get("day")
        position = next_position.get((line, day))

        if position not in positions:
            postponed[o].Start = 1
            continue

        x[o, line, day, position].Start = 1
        postponed[o].Start = 0
        next_position[(line, day)] += 1

    model.update()


def _solve_simplified_with_gurobi_legacy(instance):
    start_time = time.perf_counter()
    model = gp.Model("doceleia_scheduling")

    orders = list(range(len(instance["demand"])))
    lines = instance["final_lines"]
    days = list(range(1, instance["n_days"] + 1))

    refs_by_id = create_refs_by_id(instance)

    print("Orders:", orders)
    print("Lines:", lines)
    print("Days:", days)
    print(
        "Working-day calendar:",
        {
            day: get_calendar_date_for_day(instance, day)
            for day in days
        },
    )

    x = {}

    for o in orders:
        for l in lines:
            for d in days:
                x[o, l, d] = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"x_{o}_{l}_{d}"
                )

    delay = {}

    for o in orders:
        delay[o] = model.addVar(
            lb=0,
            vtype=GRB.CONTINUOUS,
            name=f"delay_{o}"
        )

    capacity_excess = {}

    for l in lines:
        for d in days:
            capacity_excess[l, d] = model.addVar(
                lb=0,
                vtype=GRB.CONTINUOUS,
                name=f"capacity_excess_{l}_{d}"
            )
    operator_required = {}
    for l in lines:
        for d in days: 
            operator_required[l, d] = model.addVar(
                lb=0,
                vtype=GRB.CONTINUOUS,
                name=f"operator_required_{l}_{d}"
            )
    operator_excess = {}
    for d in days: 
        operator_excess[d] = model.addVar(
            lb=0,
            vtype=GRB.CONTINUOUS,
            name=f"operator_excess_{d}"
        )
    for o in orders:
        model.addConstr(
            gp.quicksum(x[o, l, d] for l in lines for d in days) == 1,
            name=f"assign_once_{o}"
        )

    for o in orders:
        order = instance["demand"][o]
        ref = refs_by_id[str(order["ref_id"]).strip()]
        valid_lines = valid_lines_for_ref(ref)

        for l in lines:
            if l not in valid_lines:
                for d in days:
                    model.addConstr(
                        x[o, l, d] == 0,
                        name=f"incompatible_{o}_{l}_{d}"
                    )

    for o in orders:
        order = instance["demand"][o]
        delivery_date = order["delivery_date"]

        production_day = gp.quicksum(
            d * x[o, l, d]
            for l in lines
            for d in days
        )

        model.addConstr(
            delay[o] >= production_day - delivery_date,
            name=f"delay_calc_{o}"
        )

    for l in lines:
        for d in days:
            production_terms = []

            for o in orders:
                order = instance["demand"][o]
                ref = refs_by_id[str(order["ref_id"]).strip()]

                production_time = get_production_time(
                    ref,
                    l,
                    order["master_boxes"]
                )

                if production_time is not None:
                    production_terms.append(production_time * x[o, l, d])

            total_production_time = gp.quicksum(production_terms)

            model.addConstr(
                total_production_time
                <= instance["available_line_time_min"] + capacity_excess[l, d],
                name=f"capacity_{l}_{d}"
            )
    for l in lines:
        for d in days:
            for o in orders:
                order = instance["demand"][o]
                ref = refs_by_id[str(order["ref_id"]).strip()]
                valid_lines = valid_lines_for_ref(ref)

                if l in valid_lines:
                    required_ops = get_required_operators(ref, l)

                    model.addConstr(
                        operator_required[l, d] >= required_ops * x[o, l, d],
                        name=f"operator_required_{o}_{l}_{d}"
                    )

    for d in days:
        model.addConstr(
            gp.quicksum(operator_required[l, d] for l in lines)
            <= instance["standard_operators"] + operator_excess[d],
            name=f"operator_capacity_{d}"
        )


    model.setObjective(
        DELAY_WEIGHT * gp.quicksum(delay[o] for o in orders)
        + CAPACITY_WEIGHT * gp.quicksum(
            capacity_excess[l, d]
            for l in lines
            for d in days
        )
        + OPERATOR_WEIGHT * gp.quicksum(
            operator_excess[d]
            for d in days
        ),
        GRB.MINIMIZE
    )


    model.update()

    print("Number of x variables:", len(x))
    print("Number of delay variables:", len(delay))
    print("Number of capacity excess variables:", len(capacity_excess))
    print("Number of operator required variables:", len(operator_required))
    print("Number of operator excess variables:", len(operator_excess))

    model.optimize()

    if model.Status == GRB.OPTIMAL:
        delay_component = DELAY_WEIGHT * sum(delay[o].X for o in orders)
        capacity_component = CAPACITY_WEIGHT * sum(
            capacity_excess[l, d].X
            for l in lines
            for d in days
        )
        operator_component = OPERATOR_WEIGHT * sum(
            operator_excess[d].X
            for d in days
        )
        gurobi_solution = []
        print("\n=== OBJECTIVE BREAKDOWN ===")
        print(f"Total objective: {model.ObjVal:.2f}")
        print(f"Delay component: {delay_component:.2f}")
        print(f"Capacity component: {capacity_component:.2f}")
        print(f"Operator component: {operator_component:.2f}")



        print("\n=== SOLUTION ===")

        for o in orders:
            line, day = _selected_assignment(x, o, lines, days)

            if line is None:
                continue

            order = instance["demand"][o]
            gurobi_solution.append({
                "order_id": o,
                "ref_id": str(order["ref_id"]).strip(),
                "master_boxes": order["master_boxes"],
                "delivery_date": order["delivery_date"],
                "priority": order.get("priority", "Medium"),
                "day": day,
                "line": line,
            })

            ref = refs_by_id[str(order["ref_id"]).strip()]
            production_time = get_production_time(
                ref,
                line,
                order["master_boxes"]
            )
            required_ops = get_required_operators(ref, line)

            print(
                f"Order {o:02d} | ref {order['ref_id']} | "
                f"{order['master_boxes']} boxes | {line} | day {day} | "
                f"delivery {order['delivery_date']} | "
                f"delay {delay[o].X:.0f} | "
                f"time {production_time:.2f} min | "
                f"ops {required_ops}"
            )

        print("\n=== CAPACITY USAGE ===")

        for line in lines:
            for day in days:
                used_time = 0

                for o in orders:
                    if x[o, line, day].X > 0.5:
                        order = instance["demand"][o]
                        ref = refs_by_id[str(order["ref_id"]).strip()]
                        production_time = get_production_time(
                            ref,
                            line,
                            order["master_boxes"]
                        )

                        if production_time is not None:
                            used_time += production_time

                print(
                    f"{line}, day {day}: "
                    f"used {used_time:.2f} min / "
                    f"available {instance['available_line_time_min']:.2f} min / "
                    f"excess {capacity_excess[line, day].X:.2f} min"
                )

        print("\n=== OPERATORS ===")

        for d in days:
            required_by_line = {}

            for l in lines:
                line_required = 0

                for o in orders:
                    if x[o, l, d].X > 0.5:
                        order = instance["demand"][o]
                        ref = refs_by_id[str(order["ref_id"]).strip()]
                        required_ops = get_required_operators(ref, l)

                        line_required = max(line_required, required_ops)

                required_by_line[l] = line_required

            total_required = sum(required_by_line.values())
            real_excess = max(0, total_required - instance["standard_operators"])

            print(
                f"Day {d}: "
                f"required {total_required:.0f} / "
                f"standard {instance['standard_operators']} / "
                f"excess {real_excess:.0f}"
                )

            for l in lines:
                print(
                    f"  {l}: {required_by_line[l]:.0f} operators"
                )

        print("\n=== EVALUATOR CHECK ON GUROBI SOLUTION ===")
        evaluator_metrics = evaluate_solution(gurobi_solution, instance)
        evaluator_metrics["computation_time_sec"] = time.perf_counter() - start_time
        print_metrics(evaluator_metrics)
        print("\n=== CAPACITY EXCESS ONLY ===")

        for l in lines:
            for d in days:
                if capacity_excess[l, d].X > 0.001:
                    print(
                        f"Line {l}, day {d}: "
                        f"excess {capacity_excess[l, d].X:.2f} min"
                    )

        return gurobi_solution, evaluator_metrics

    return None, None


def solve_with_gurobi(instance, time_limit=1800, verbose=True):
    start_time = time.perf_counter()
    model = gp.Model("doceleia_full_small")

    if not verbose:
        model.Params.OutputFlag = 0

    model.Params.TimeLimit = time_limit

    orders = list(range(len(instance["demand"])))
    lines = instance["final_lines"]
    days = list(range(1, instance["n_days"] + 1))
    positions = list(range(1, len(orders) + 1))
    refs_by_id = create_refs_by_id(instance)
    max_values = compute_max_values(instance)

    x = model.addVars(
        orders,
        lines,
        days,
        positions,
        vtype=GRB.BINARY,
        name="x"
    )
    postponed = model.addVars(
        orders,
        vtype=GRB.BINARY,
        name="postponed"
    )
    delay = model.addVars(
        orders,
        lb=0,
        vtype=GRB.CONTINUOUS,
        name="delay"
    )
    pair = {}

    for i in orders:
        for j in orders:
            if i == j:
                continue

            for l in lines:
                for d in days:
                    for p in positions[1:]:
                        pair[i, j, l, d, p] = model.addVar(
                            vtype=GRB.BINARY,
                            name=f"pair_{i}_{j}_{l}_{d}_{p}"
                        )

    model.update()

    for o in orders:
        model.addConstr(
            gp.quicksum(
                x[o, l, d, p]
                for l in lines
                for d in days
                for p in positions
            ) + postponed[o] == 1,
            name=f"assign_or_postpone_{o}"
        )

    for o in orders:
        order = instance["demand"][o]
        ref = refs_by_id[str(order["ref_id"]).strip()]
        valid_lines = valid_lines_for_ref(ref)
        valid_days = get_valid_days_for_ref(instance, ref)

        for l in lines:
            for d in days:
                if l not in valid_lines or d not in valid_days:
                    for p in positions:
                        model.addConstr(
                            x[o, l, d, p] == 0,
                            name=f"infeasible_{o}_{l}_{d}_{p}"
                        )

    for l in lines:
        for d in days:
            for p in positions:
                model.addConstr(
                    gp.quicksum(x[o, l, d, p] for o in orders) <= 1,
                    name=f"one_order_per_position_{l}_{d}_{p}"
                )

            for p in positions[:-1]:
                model.addConstr(
                    gp.quicksum(x[o, l, d, p + 1] for o in orders)
                    <= gp.quicksum(x[o, l, d, p] for o in orders),
                    name=f"no_gaps_{l}_{d}_{p}"
                )

    for o in orders:
        order = instance["demand"][o]
        delivery_date = order["delivery_date"]

        production_day = gp.quicksum(
            d * x[o, l, d, p]
            for l in lines
            for d in days
            for p in positions
        )

        model.addConstr(
            delay[o] >= production_day - delivery_date,
            name=f"delay_{o}"
        )

    for i in orders:
        for j in orders:
            if i == j:
                continue

            for l in lines:
                for d in days:
                    for p in positions[1:]:
                        model.addConstr(
                            pair[i, j, l, d, p] <= x[i, l, d, p - 1],
                            name=f"pair_prev_{i}_{j}_{l}_{d}_{p}"
                        )
                        model.addConstr(
                            pair[i, j, l, d, p] <= x[j, l, d, p],
                            name=f"pair_curr_{i}_{j}_{l}_{d}_{p}"
                        )
                        model.addConstr(
                            pair[i, j, l, d, p]
                            >= x[i, l, d, p - 1] + x[j, l, d, p] - 1,
                            name=f"pair_link_{i}_{j}_{l}_{d}_{p}"
                        )

    setup_expr = gp.LinExpr()
    capacity_utilisation_expr = gp.LinExpr()
    capacity_slots = len(lines) * len(days)

    for l in lines:
        for d in days:
            production_terms = []
            setup_terms = []

            for o in orders:
                order = instance["demand"][o]
                ref = refs_by_id[str(order["ref_id"]).strip()]
                production_time = evaluator_get_production_time(
                    ref,
                    l,
                    order["master_boxes"]
                )

                if production_time is not None:
                    for p in positions:
                        production_terms.append(production_time * x[o, l, d, p])

            for i in orders:
                ref_i = refs_by_id[str(instance["demand"][i]["ref_id"]).strip()]

                for j in orders:
                    if i == j:
                        continue

                    ref_j = refs_by_id[str(instance["demand"][j]["ref_id"]).strip()]
                    setup_time = get_setup(
                        instance,
                        ref_i["family"],
                        ref_j["family"]
                    )

                    for p in positions[1:]:
                        term = setup_time * pair[i, j, l, d, p]
                        setup_terms.append(term)
                        setup_expr += term

            production_expr = gp.quicksum(production_terms)
            available_capacity = get_available_line_time_for_day(instance, d)

            model.addConstr(
                production_expr
                + gp.quicksum(setup_terms)
                <= (
                    available_capacity
                    + get_capacity_tolerance_for_day(instance, d)
                ),
                name=f"capacity_{l}_{d}"
            )

            capacity_used = model.addVar(
                lb=0,
                ub=available_capacity,
                vtype=GRB.CONTINUOUS,
                name=f"capacity_used_{l}_{d}",
            )
            model.addConstr(
                capacity_used <= production_expr,
                name=f"capacity_used_link_{l}_{d}",
            )

            if available_capacity > 0 and capacity_slots > 0:
                capacity_utilisation_expr += (
                    capacity_used
                    / available_capacity
                    / capacity_slots
                )

    postponed_volume_expr = gp.LinExpr()
    scheduled_economic_value_expr = gp.LinExpr()
    operator_usage_expr = gp.LinExpr()

    for o in orders:
        order = instance["demand"][o]
        ref = refs_by_id[str(order["ref_id"]).strip()]
        master_boxes = order["master_boxes"]
        economic_value = (
            master_boxes
            * (ref.get("economic_value_per_master_box") or 0)
        )

        scheduled_expr = gp.quicksum(
            x[o, l, d, p]
            for l in lines
            for d in days
            for p in positions
        )
        postponed_volume_expr += master_boxes * postponed[o]
        scheduled_economic_value_expr += economic_value * scheduled_expr

        for l in lines:
            production_time = evaluator_get_production_time(
                ref,
                l,
                master_boxes,
            )

            if production_time is None:
                continue

            production_operator_minutes = (
                production_time * get_production_operators(ref, l)
            )
            finishing_operator_minutes = 0

            if l == "L1":
                finishing_operator_minutes = (
                    max(0, production_time - FINISHING_DELAY_L1_MIN)
                    * get_finishing_operators(ref, l)
                )
            elif l == "L2":
                finishing_operator_minutes = (
                    production_time
                    * get_finishing_operators(ref, l)
                )

            operator_usage_expr += (
                production_operator_minutes
                + finishing_operator_minutes
            ) * gp.quicksum(
                x[o, l, d, p]
                for d in days
                for p in positions
            )

    for i in orders:
        ref_i = refs_by_id[str(instance["demand"][i]["ref_id"]).strip()]

        for j in orders:
            if i == j:
                continue

            ref_j = refs_by_id[str(instance["demand"][j]["ref_id"]).strip()]
            setup_time = get_setup(
                instance,
                ref_i["family"],
                ref_j["family"],
            )

            for l in lines:
                setup_operators = get_production_operators(ref_i, l)

                for d in days:
                    for p in positions[1:]:
                        operator_usage_expr += (
                            setup_time
                            * setup_operators
                            * pair[i, j, l, d, p]
                        )

    objective = (
        POSTPONEMENT_NORMALIZED_WEIGHT
        * postponed_volume_expr
        / max_values["postponed_volume"]
        + DELAY_NORMALIZED_WEIGHT
        * gp.quicksum(delay[o] for o in orders)
        / max_values["delay_days"]
        + SETUP_NORMALIZED_WEIGHT
        * setup_expr
        / max_values["setup_time"]
        - ECONOMIC_VALUE_NORMALIZED_WEIGHT
        * scheduled_economic_value_expr
        / max_values["economic_value"]
        - CAPACITY_UTILIZATION_NORMALIZED_WEIGHT
        * capacity_utilisation_expr
        - OPERATOR_UTILIZATION_NORMALIZED_WEIGHT
        * operator_usage_expr
        / max_values["operator_minutes"]
    )

    model.setObjective(objective, GRB.MINIMIZE)
    apply_feasible_mip_start(
        model,
        instance,
        x,
        postponed,
        orders,
        lines,
        days,
        positions,
    )

    print("Full Gurobi model")
    print("Orders/lots:", len(orders))
    print("Lines:", lines)
    print("Days:", days)
    print("Positions:", len(positions))

    operator_cuts = 0
    solution = None
    metrics = None

    while True:
        elapsed = time.perf_counter() - start_time
        remaining_time = max(1, time_limit - elapsed)
        model.Params.TimeLimit = remaining_time
        model.optimize()

        if model.SolCount == 0:
            raise RuntimeError(
                f"Gurobi did not find a solution. Status: {model.Status}"
            )

        candidate_solution, selected_variables = extract_full_solution(
            instance,
            x,
            postponed,
            orders,
            lines,
            days,
            positions,
        )
        candidate_metrics = evaluate_solution(candidate_solution, instance)

        if candidate_metrics["operator_violations"] == 0:
            solution = candidate_solution
            metrics = candidate_metrics
            break

        operator_cuts += 1
        model.addConstr(
            gp.quicksum(selected_variables) <= len(selected_variables) - 1,
            name=f"operator_feasibility_cut_{operator_cuts}",
        )

        if verbose:
            print(
                f"Operator feasibility cut {operator_cuts}: "
                f"{candidate_metrics['operator_violations']} violating slots",
                flush=True,
            )

        if time.perf_counter() - start_time >= time_limit:
            raise RuntimeError(
                "Time limit reached before finding a solution that respects "
                "the hourly operator constraint."
            )

    breakdown = normalised_fitness_breakdown(metrics, max_values)
    metrics["normalised_fitness"] = normalised_fitness(metrics, max_values)
    metrics["normalised_fitness_breakdown"] = breakdown
    metrics["max_values"] = max_values
    metrics["gurobi_objective"] = model.ObjVal
    metrics["objective_evaluator_difference"] = (
        model.ObjVal - metrics["normalised_fitness"]
    )
    metrics["computation_time_sec"] = time.perf_counter() - start_time

    status_name = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
    }.get(model.Status, str(model.Status))

    print("\n=== FULL GUROBI SUMMARY ===")
    print(f"Solver status: {status_name}")
    print(f"Gurobi objective: {model.ObjVal:.6f}")
    print(f"Evaluator normalised fitness: {metrics['normalised_fitness']:.6f}")
    print(
        "Objective/evaluator difference: "
        f"{metrics['objective_evaluator_difference']:+.6f}"
    )
    print(f"Gurobi best bound: {model.ObjBound:.6f}")
    print(f"Gurobi MIP gap: {model.MIPGap * 100:.2f}%")
    print(f"Operator feasibility cuts: {operator_cuts}")
    print("\n=== NORMALISED FITNESS BREAKDOWN ===")
    print(f"Postponement contribution: {breakdown['postponement']:+.6f}")
    print(f"Delay contribution: {breakdown['delay']:+.6f}")
    print(f"Setup contribution: {breakdown['setup']:+.6f}")
    print(f"Economic value contribution: {breakdown['economic_value']:+.6f}")
    print(
        "Capacity utilisation contribution: "
        f"{breakdown['capacity_utilisation']:+.6f}"
    )
    print(
        "Operator utilisation contribution: "
        f"{breakdown['operator_utilisation']:+.6f}"
    )
    print(f"Normalised fitness Z: {breakdown['total']:+.6f}")
    print_metrics(metrics)

    return solution, metrics, {
        "objective": model.ObjVal,
        "best_bound": model.ObjBound,
        "mip_gap": model.MIPGap,
        "status": model.Status,
        "status_name": status_name,
        "operator_feasibility_cuts": operator_cuts,
    }


def solve_simplified_with_gurobi(instance, time_limit=1800, verbose=True):
    """Compatibility entry point using the current official model."""
    return solve_with_gurobi(
        instance,
        time_limit=time_limit,
        verbose=verbose,
    )


if __name__ == "__main__":
    instance = load_real_instance(
        "../Inputs_EmpresaX_small.xlsx",
        operational_config={
            "shift_start_min": 480,
            "shift_end_min": 990,
            "lunch_break_min": 30,
            "cleaning_time_min": 30,
        },
    )

    solution, metrics, info = solve_with_gurobi(
        instance,
        time_limit=1800,
        verbose=True,
    )

    print("\n=== BEST GUROBI SCHEDULE FOUND ===")
    for item in solution:
        if item.get("postponed"):
            status = "POSTPONED"
        else:
            calendar_date = get_calendar_date_for_day(
                instance,
                item["day"],
            )
            status = (
                f"day {item['day']} ({calendar_date}) | {item['line']}"
            )

        print(
            f"{item['order_id']:02d}. "
            f"{item['ref_id']} | "
            f"{item['master_boxes']} boxes | "
            f"{status} | "
            f"delivery {item['delivery_date']}"
        )

    print("\n=== GUROBI METRICS FOR COMPARISON ===")
    print(f"Computation time: {metrics['computation_time_sec']:.2f} sec")
    print(f"Normalised fitness: {metrics['normalised_fitness']:.6f}")
    print(f"Scheduled economic value: {metrics['scheduled_economic_value']:.2f}")
    print(f"Postponed economic value: {metrics['postponed_economic_value']:.2f}")
    print(f"Total capacity excess: {metrics['total_capacity_excess']:.2f} min")
    print(f"Total setup time: {metrics['setup_total_min']:.2f} min")
    print(f"Total delay: {metrics['delay_days_total']} days")
    print(f"Number of postponed orders: {metrics['postponed_orders']}")
    print(f"Gurobi objective: {info['objective']:.2f}")
    print(f"Gurobi best bound: {info['best_bound']:.2f}")
    print(f"Gurobi MIP gap: {info['mip_gap'] * 100:.2f}%")

    print_validation_report(
        solution,
        instance,
        title="GUROBI VALIDATION REPORT"
    )
