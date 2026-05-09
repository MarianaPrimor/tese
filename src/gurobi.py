import gurobipy as gp
from gurobipy import GRB

from generate_instance import load_real_instance, calculate_production_time
from evaluator import create_refs_by_id, valid_lines_for_ref


DELAY_WEIGHT = 100
CAPACITY_WEIGHT = 50


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
            total_production_time = gp.quicksum(
                get_production_time(
                    refs_by_id[str(instance["demand"][o]["ref_id"]).strip()],
                    l,
                    instance["demand"][o]["master_boxes"]
                ) * x[o, l, d]
                for o in orders
            )

            model.addConstr(
                total_production_time
                <= instance["available_line_time_min"] + capacity_excess[l, d],
                name=f"capacity_{l}_{d}"
            )

    model.setObjective(
        DELAY_WEIGHT * gp.quicksum(delay[o] for o in orders)
        + CAPACITY_WEIGHT * gp.quicksum(
            capacity_excess[l, d]
            for l in lines
            for d in days
        ),
        GRB.MINIMIZE
    )

    model.update()

    print("Number of x variables:", len(x))
    print("Number of delay variables:", len(delay))
    print("Number of capacity excess variables:", len(capacity_excess))

    model.optimize()

    if model.Status == GRB.OPTIMAL:
        print("\n=== SOLUTION ===")

        for o in orders:
            for l in lines:
                for d in days:
                    if x[o, l, d].X > 0.5:
                        print(
                            f"Order {o} -> line {l}, day {d}, "
                            f"delivery {instance['demand'][o]['delivery_date']}, "
                            f"delay {delay[o].X}"
                        )

        print("\n=== CAPACITY EXCESS ===")

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
