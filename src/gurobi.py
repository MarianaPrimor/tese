import gurobipy as gp
from gurobipy import GRB

from generate_instance import load_real_instance, calculate_production_time
from evaluator import create_refs_by_id, valid_lines_for_ref


DELAY_WEIGHT = 100
CAPACITY_WEIGHT = 50
OPERATOR_WEIGHT = 80


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


def _selected_assignment(x, order, lines, days):
    for line in lines:
        for day in days:
            if x[order, line, day].X > 0.5:
                return line, day

    return None, None


def solve_with_gurobi(instance):
    model = gp.Model("doceleia_scheduling")

    orders = list(range(len(instance["demand"])))
    lines = instance["final_lines"]
    days = list(range(1, instance["n_days"] + 1))

    refs_by_id = create_refs_by_id(instance)

    print("Orders:", orders)
    print("Lines:", lines)
    print("Days:", days)

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

        print("\n=== OBJECTIVE BREAKDOWN ===")
        print(f"Total objective: {model.ObjVal:.2f}")
        print(f"Delay penalty: {delay_component:.2f}")
        print(f"Capacity penalty: {capacity_component:.2f}")
        print(f"Operator penalty: {operator_component:.2f}")

        print("\n=== SOLUTION ===")

        for o in orders:
            line, day = _selected_assignment(x, o, lines, days)

            if line is None:
                continue

            order = instance["demand"][o]
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
            total_required = sum(operator_required[l, d].X for l in lines)

            print(
                f"Day {d}: "
                f"required {total_required:.0f} / "
                f"standard {instance['standard_operators']} / "
                f"excess {operator_excess[d].X:.0f}"
            )

            for l in lines:
                print(
                    f"  {l}: {operator_required[l, d].X:.0f} operators"
                )

        print("\n=== CAPACITY EXCESS ONLY ===")

        for l in lines:
            for d in days:
                if capacity_excess[l, d].X > 0.001:
                    print(
                        f"Line {l}, day {d}: "
                        f"excess {capacity_excess[l, d].X:.2f} min"
                    )


if __name__ == "__main__":
    instance = load_real_instance("../Inputs_Doceleia.xlsx")
    solve_with_gurobi(instance)
