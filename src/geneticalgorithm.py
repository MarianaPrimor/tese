from copy import deepcopy
import random
import time

from generate_instance import load_real_instance
from evaluator import (
    generate_random_solution,
    evaluate_solution,
    print_solution,
    print_metrics,
    create_refs_by_id,
    valid_lines_for_ref,
    get_valid_days_for_ref,
    get_setup,
    get_production_time,
    get_available_line_time_for_day,
    get_order_economic_value,
    simulate_time_schedule,
    calculate_operator_usage_exact,
    get_capacity_tolerance_for_day,
    DEFAULT_NORMALISED_WEIGHTS,
    compute_max_values,
    normalised_fitness,
    normalised_fitness_breakdown,
)


# GENERATING INITIAL POPULATION


def generate_initial_population(
    instance,
    population_size=200,
    seed=0,
    objective_weights=None,
    heuristic_ratio=0.0,
):
    population = []
    heuristic_count = int(round(population_size * max(0.0, min(1.0, heuristic_ratio))))

    for i in range(population_size):
        solution_seed = seed * 100000 + i
        if i < heuristic_count:
            solution = generate_edd_solution(instance, seed=solution_seed)
        else:
            solution = generate_random_solution(instance, seed=solution_seed)
        solution = enforce_hard_constraints(
            solution,
            instance,
            objective_weights=objective_weights,
        )
        population.append(solution)

    return population


def generate_edd_solution(instance, seed=42):
    rng = random.Random(seed)
    refs_by_id = create_refs_by_id(instance)
    decorated_orders = []
    skipped_orders = 0

    for order_index, order in enumerate(instance["demand"]):
        ref_id = str(order["ref_id"]).strip()

        if ref_id not in refs_by_id:
            print(
                f"WARNING: demand ref_id {ref_id} not found in references. "
                f"Skipping order."
            )
            skipped_orders += 1
            continue

        ref = refs_by_id[ref_id]
        economic_value = get_order_economic_value(order, ref)
        delivery_date = order.get("delivery_date")
        sort_delivery = (
            delivery_date
            if delivery_date is not None
            else instance.get("n_days", 0) + 1
        )
        decorated_orders.append((
            sort_delivery,
            -economic_value,
            rng.random(),
            order_index,
            order,
            ref,
            ref_id,
        ))

    decorated_orders.sort()

    solution = []

    for _, _, _, order_index, order, ref, ref_id in decorated_orders:
        valid_lines = valid_lines_for_ref(ref)
        valid_days = get_valid_days_for_ref(instance, ref)

        if not valid_lines or not valid_days:
            line = None
            day = None
            postponed = True
        else:
            line = rng.choice(valid_lines)
            day = rng.choice(valid_days)
            postponed = False

        solution.append({
            "order_id": order_index,
            "ref_id": ref_id,
            "master_boxes": order["master_boxes"],
            "delivery_date": order["delivery_date"],
            "delivery_calendar_date": order.get("delivery_calendar_date"),
            "adjusted_delivery_date": order.get("adjusted_delivery_date"),
            "day": day,
            "line": line,
            "postponed": postponed,
        })

    if skipped_orders > 0:
        print(f"WARNING: skipped {skipped_orders} demand orders.")

    return solution



# FITNESS


def fitness(solution, instance, max_values, objective_weights=None):
    metrics = evaluate_solution(solution, instance)
    return normalised_fitness(
        metrics,
        max_values,
        weights=objective_weights,
    )


def evaluate_with_normalised_fitness(
    solution,
    instance,
    max_values,
    objective_weights=None,
):
    metrics = evaluate_solution(solution, instance)
    breakdown = normalised_fitness_breakdown(
        metrics,
        max_values,
        weights=objective_weights,
    )
    score = breakdown["total"]

    metrics["normalised_fitness"] = score
    metrics["normalised_fitness_breakdown"] = breakdown
    metrics["max_values"] = max_values
    return metrics


def choose_feasible_day(gene, valid_days):
    delivery_date = gene.get("delivery_date")

    if delivery_date is not None:
        on_time_days = [
            day for day in valid_days
            if day <= delivery_date
        ]

        if on_time_days:
            return min(on_time_days)

    return min(valid_days)


def enforce_hard_constraints(
    solution,
    instance,
    objective_weights=None,
    reinsert_postponed=False,
):
    refs_by_id = create_refs_by_id(instance)
    repaired_solution = deepcopy(solution)
    max_values = compute_max_values(instance)
    weights = (
        DEFAULT_NORMALISED_WEIGHTS
        if objective_weights is None
        else objective_weights
    )

    for gene in repaired_solution:
        ref_id = str(gene.get("ref_id")).strip()
        ref = refs_by_id.get(ref_id)

        if ref is None:
            gene["line"] = None
            gene["day"] = None
            gene["postponed"] = True
            continue

        valid_lines = valid_lines_for_ref(ref)
        valid_days = get_valid_days_for_ref(instance, ref)

        if gene.get("postponed") or not valid_lines or not valid_days:
            gene["line"] = None
            gene["day"] = None
            gene["postponed"] = True
            continue

        if gene.get("line") not in valid_lines:
            gene["line"] = valid_lines[0]

        if gene.get("day") not in valid_days:
            gene["day"] = choose_feasible_day(gene, valid_days)

        gene["postponed"] = False

    repaired_solution = repair_capacity_constraints(
        repaired_solution,
        instance,
        max_values,
        weights,
    )
    repaired_solution = repair_operator_constraints(
        repaired_solution,
        instance,
        max_values,
        weights,
    )

    if reinsert_postponed:
        repaired_solution = reinsert_postponed_orders(
            repaired_solution,
            instance,
            max_values,
            weights,
        )

    return repaired_solution


# ============================================================
# SELECTION
# ============================================================

def tournament_selection(
    population,
    instance,
    max_values,
    tournament_size=3,
    objective_weights=None,
):
    candidates = random.sample(population, tournament_size)

    best_candidate = min(
        candidates,
        key=lambda solution: fitness(
            solution,
            instance,
            max_values,
            objective_weights=objective_weights,
        )
    )

    return deepcopy(best_candidate)


# ============================================================
# CROSSOVER
# ============================================================

def crossover(parent_1, parent_2):
    size = len(parent_1)

    start, end = sorted(random.sample(range(size), 2))

    child = [None] * size

    copied_order_ids = set()

    for i in range(start, end + 1):
        child[i] = deepcopy(parent_1[i])
        copied_order_ids.add(parent_1[i]["order_id"])
    
    parent_2_index = 0

    for i in range(size):
        if child[i] is not None:
            continue

        while parent_2[parent_2_index]["order_id"] in copied_order_ids:
            parent_2_index += 1

        child[i] = deepcopy(parent_2[parent_2_index])
        copied_order_ids.add(parent_2[parent_2_index]["order_id"])
       
    return child


# ============================================================
# MUTATION
# ============================================================

def mutate(
    solution,
    instance,
    mutation_rate=0.057,
):
    mutated_solution = deepcopy(solution)

    # Swap mutation: exchanges the position of two orders
    if random.random() < mutation_rate and len(mutated_solution) >= 2:
        i, j = random.sample(range(len(mutated_solution)), 2)

        mutated_solution[i], mutated_solution[j] = (
            mutated_solution[j],
            mutated_solution[i]
        )

    # Insertion mutation: removes one order and inserts it elsewhere
    if random.random() < mutation_rate and len(mutated_solution) >= 2:
        i, j = random.sample(range(len(mutated_solution)), 2)

        gene = mutated_solution.pop(i)
        mutated_solution.insert(j, gene)

    return mutated_solution


def get_economic_value_for_repair(gene, ref):
    return (
        gene.get("master_boxes", 0)
        * (ref.get("economic_value_per_master_box") or 0)
    )


def calculate_group_time_for_repair(genes, refs_by_id, instance, line):
    total_time = 0
    previous_family = None
    timed_genes = []

    for gene in genes:
        ref_id = str(gene["ref_id"]).strip()
        ref = refs_by_id.get(ref_id)

        if ref is None:
            continue

        production_time = get_production_time(
            ref,
            line,
            gene["master_boxes"]
        )

        if production_time is None:
            continue

        setup_time = get_setup(
            instance,
            previous_family,
            ref["family"]
        )

        total_time += production_time + setup_time
        previous_family = ref["family"]
        timed_genes.append((gene, ref, production_time + setup_time))

    return total_time, timed_genes


def postpone_gene(gene):
    gene["line"] = None
    gene["day"] = None
    gene["postponed"] = True


def schedule_gene(gene, day, line):
    gene["day"] = day
    gene["line"] = line
    gene["postponed"] = False


def get_postponement_objective_cost(gene, ref, max_values, weights):
    economic_value = get_order_economic_value(gene, ref)
    day = gene.get("day")
    due_day = gene.get("delivery_date")
    avoided_delay_ratio = 0

    if day is not None and due_day is not None and day > due_day:
        avoided_delay_ratio = (day - due_day) / max_values["delay_days"]

    postponed_ratio = (
        gene.get("master_boxes", 0)
        / max_values["postponed_volume"]
    )
    economic_ratio = economic_value / max_values["economic_value"]

    return max(
        1e-9,
        weights["postponement"] * postponed_ratio
        + weights["economic_value"] * economic_ratio
        - weights["delay"] * avoided_delay_ratio,
    )


def repair_capacity_constraints(solution, instance, max_values, weights):
    refs_by_id = create_refs_by_id(instance)
    repaired_solution = deepcopy(solution)
    groups = {}

    for gene in repaired_solution:
        if gene.get("postponed"):
            continue

        key = (gene.get("day"), gene.get("line"))
        groups.setdefault(key, []).append(gene)

    for (day, line), genes in groups.items():
        if day is None or line not in instance["final_lines"]:
            continue

        capacity_limit = (
            get_available_line_time_for_day(instance, day)
            + get_capacity_tolerance_for_day(instance, day)
        )

        while genes:
            total_time, _ = calculate_group_time_for_repair(
                genes,
                refs_by_id,
                instance,
                line,
            )

            if total_time <= capacity_limit:
                break

            candidates = []

            for index, gene in enumerate(genes):
                ref_id = str(gene.get("ref_id")).strip()
                ref = refs_by_id.get(ref_id)

                if ref is None:
                    candidates.append((0, index))
                    continue

                remaining_genes = genes[:index] + genes[index + 1:]
                remaining_time, _ = calculate_group_time_for_repair(
                    remaining_genes,
                    refs_by_id,
                    instance,
                    line,
                )
                freed_time = max(1, total_time - remaining_time)
                postponement_cost = get_postponement_objective_cost(
                    gene,
                    ref,
                    max_values,
                    weights,
                )
                candidates.append((
                    postponement_cost / freed_time,
                    index,
                ))

            _, selected_index = min(candidates)
            selected_gene = genes.pop(selected_index)
            postpone_gene(selected_gene)

    return repaired_solution


def repair_operator_constraints(solution, instance, max_values, weights):
    refs_by_id = create_refs_by_id(instance)
    repaired_solution = deepcopy(solution)
    max_iterations = len(repaired_solution)

    for _ in range(max_iterations):
        operations = simulate_time_schedule(repaired_solution, instance)
        exact_operator_usage = calculate_operator_usage_exact(
            operations,
            instance,
        )
        violating_intervals = [
            interval
            for interval in exact_operator_usage["intervals"]
            if interval["excess"] > 0
        ]

        if not violating_intervals:
            break

        relief_by_order = {}

        for operation in operations:
            order_id = operation.get("order_id")

            if order_id is None or operation["end"] <= operation["start"]:
                continue

            for interval in violating_intervals:
                if operation["day"] != interval["day"]:
                    continue

                overlap = (
                    min(operation["end"], interval["end"])
                    - max(operation["start"], interval["start"])
                )

                if overlap <= 0:
                    continue

                relief_by_order[order_id] = (
                    relief_by_order.get(order_id, 0)
                    + min(
                        operation.get("operators", 0),
                        interval["excess"],
                    )
                    * overlap
                )

        candidates = []

        for gene in repaired_solution:
            order_id = gene.get("order_id")
            relief = relief_by_order.get(order_id, 0)

            if gene.get("postponed") or relief <= 0:
                continue

            ref = refs_by_id.get(str(gene.get("ref_id")).strip())

            if ref is None:
                cost = 0
            else:
                cost = get_postponement_objective_cost(
                    gene,
                    ref,
                    max_values,
                    weights,
                )

            candidates.append((cost / relief, order_id, gene))

        if not candidates:
            break

        _, _, selected_gene = min(
            candidates,
            key=lambda candidate: (candidate[0], candidate[1]),
        )
        postpone_gene(selected_gene)

    return repaired_solution


def reinsert_postponed_orders(solution, instance, max_values, weights):
    refs_by_id = create_refs_by_id(instance)
    repaired_solution = deepcopy(solution)
    current_score = normalised_fitness(
        evaluate_solution(repaired_solution, instance),
        max_values,
        weights=weights,
    )

    inserted = True

    while inserted:
        inserted = False
        best_candidate = None
        best_score = current_score

        postponed_genes = sorted(
            (
                gene for gene in repaired_solution
                if gene.get("postponed")
            ),
            key=lambda gene: (
                gene.get("delivery_date") or instance["n_days"] + 1,
                gene.get("order_id"),
            ),
        )

        for gene in postponed_genes:
            ref = refs_by_id.get(str(gene.get("ref_id")).strip())

            if ref is None:
                continue

            for day in get_valid_days_for_ref(instance, ref):
                for line in valid_lines_for_ref(ref):
                    candidate = deepcopy(repaired_solution)
                    candidate_gene = next(
                        item for item in candidate
                        if item.get("order_id") == gene.get("order_id")
                    )
                    schedule_gene(candidate_gene, day, line)

                    candidate_metrics = evaluate_solution(
                        candidate,
                        instance,
                    )

                    if candidate_metrics.get("infeasible_solution"):
                        continue

                    candidate_score = normalised_fitness(
                        candidate_metrics,
                        max_values,
                        weights=weights,
                    )

                    if candidate_score < best_score:
                        best_score = candidate_score
                        best_candidate = candidate

        if best_candidate is not None:
            repaired_solution = best_candidate
            current_score = best_score
            inserted = True

    return repaired_solution

# ============================================================
# GENETIC ALGORITHM
# ============================================================

def run_genetic_algorithm(
    instance,
    population_size=108,
    generations=200,
    mutation_rate=0.057,
    elite_size=5,
    tournament_size=3,
    stagnation_k=26,
    seed=0,
    verbose= True,
    objective_weights=None,
    heuristic_ratio=0.0,
    return_history=False,
):
    start_time = time.perf_counter()
    random.seed(seed)
    max_values = compute_max_values(instance)

    if verbose:
        print(
            f"[GA] Generating initial population ({population_size} solutions)...",
            flush=True,
        )

    population = generate_initial_population(
        instance,
        population_size,
        seed=seed,
        objective_weights=objective_weights,
        heuristic_ratio=heuristic_ratio,
    )

    if verbose:
        print("[GA] Evaluating initial population...", flush=True)

    random.seed(seed)
    population = sorted(
        population,
        key=lambda solution: fitness(
            solution,
            instance,
            max_values,
            objective_weights=objective_weights,
        )
    )
    initial_best_solution = deepcopy(population[0])
    initial_best_metrics = evaluate_with_normalised_fitness(
        initial_best_solution,
        instance,
        max_values,
        objective_weights=objective_weights,
    )

    best_solution = deepcopy(initial_best_solution)
    best_metrics = initial_best_metrics
    generations_without_improvement = 0
    actual_generations = 0
    convergence_history = [{
        "generation": 0,
        "best_fitness": best_metrics["normalised_fitness"],
    }]
    
    for generation in range(1, generations + 1):
        actual_generations = generation
        population = sorted(
            population,
            key=lambda solution: fitness(
                solution,
                instance,
                max_values,
                objective_weights=objective_weights,
            )
        )

        current_best = population[0]
        current_metrics = evaluate_with_normalised_fitness(
            current_best,
            instance,
            max_values,
            objective_weights=objective_weights,
        )
        if current_metrics["normalised_fitness"] < best_metrics["normalised_fitness"]:
            best_solution = deepcopy(current_best)
            best_metrics = current_metrics
            generations_without_improvement = 0  # reset
        else:
            generations_without_improvement += 1  # increment

        convergence_history.append({
            "generation": generation,
            "best_fitness": best_metrics["normalised_fitness"],
        })

        # Stagnation check
        if generations_without_improvement >= stagnation_k:
            if verbose:
                print(
                    f"[EARLY STOP] Stagnation at generation {generation}",
                    flush=True,
                )
            break

        if (
            best_metrics is None
            or current_metrics["normalised_fitness"] < best_metrics["normalised_fitness"]
        ):
            best_solution = deepcopy(current_best)
            best_metrics = current_metrics

        if verbose:
            print(
                f"Generation {generation:03d} | "
                f"best fitness: {best_metrics['normalised_fitness']:.6f} | "
                f"current best: {current_metrics['normalised_fitness']:.6f}",
                flush=True,
            )


        new_population = [
            enforce_hard_constraints(
                solution,
                instance,
                objective_weights=objective_weights,
            )
            for solution in population[:elite_size]
        ]

        while len(new_population) < population_size:
            parent_1 = tournament_selection(
                population,
                instance,
                max_values,
                tournament_size=tournament_size,
                objective_weights=objective_weights,
            )

            parent_2 = tournament_selection(
                population,
                instance,
                max_values,
                tournament_size=tournament_size,
                objective_weights=objective_weights,
            )

            child = crossover(parent_1, parent_2)
            child = mutate(
                child,
                instance,
                mutation_rate=mutation_rate,
            )

            child = enforce_hard_constraints(
                child,
                instance,
                objective_weights=objective_weights,
            )

            new_population.append(child)

        population = new_population

    best_solution = enforce_hard_constraints(
        best_solution,
        instance,
        objective_weights=objective_weights,
        reinsert_postponed=True,
    )
    best_metrics = evaluate_with_normalised_fitness(
        best_solution,
        instance,
        max_values,
        objective_weights=objective_weights,
    )
    best_metrics["computation_time_sec"] = time.perf_counter() - start_time

    initial_fitness = initial_best_metrics["normalised_fitness"]
    final_fitness = best_metrics["normalised_fitness"]

    if initial_fitness != 0:
        improvement = (
            (initial_fitness - final_fitness)
            / abs(initial_fitness)
        ) * 100
    else: 
        improvement = 0.0

    print("\n=== GENETIC ALGORITHM SUMMARY ===")
    print(f"Initial best fitness: {initial_fitness:.6f}")
    print(f"Final best fitness: {final_fitness:.6f}")
    print(f"Improvement: {improvement:.2f}%")
    print(f"Initial delay: {initial_best_metrics['delay_days_total']} days")
    print(f"Final delay: {best_metrics['delay_days_total']} days")
    print(f"Initial capacity excess: {initial_best_metrics['total_capacity_excess']:.2f} min")
    print(f"Final capacity excess: {best_metrics['total_capacity_excess']:.2f} min")
    print(f"Initial operator excess: {initial_best_metrics['total_operator_excess']:.2f}")
    print(f"Final operator excess: {best_metrics['total_operator_excess']:.2f}")
    print(f"Initial setup time: {initial_best_metrics['setup_total_min']:.2f} min")
    print(f"Final setup time: {best_metrics['setup_total_min']:.2f} min")
    print("\n=== NORMALISED FITNESS BREAKDOWN ===")
    breakdown = best_metrics["normalised_fitness_breakdown"]
    print(f"Postponement contribution: {breakdown['postponement']:+.6f}")
    print(f"Delay contribution: {breakdown['delay']:+.6f}")
    print(f"Setup contribution: {breakdown['setup']:+.6f}")
    print(f"Economic value contribution: {breakdown['economic_value']:+.6f}")
    print(
        "Capacity utilisation contribution: "
        f"{breakdown.get('capacity_utilisation', 0):+.6f}"
    )
    print(
        "Operator utilisation contribution: "
        f"{breakdown['operator_utilisation']:+.6f}"
    )
    print(f"Normalised fitness Z: {breakdown['total']:+.6f}")
    print(f"Maximum postponed volume: {max_values['postponed_volume']:.2f}")
    print(f"Maximum delay days: {max_values['delay_days']:.2f}")
    print(f"Maximum setup time: {max_values['setup_time']:.2f}")
    print(f"Maximum economic value: {max_values['economic_value']:.2f}")
    print(f"Maximum capacity time: {max_values['capacity_time']:.2f}")
    print(f"Maximum operator minutes: {max_values['operator_minutes']:.2f}")

    if return_history:
        return best_solution, best_metrics, actual_generations, convergence_history

    return best_solution, best_metrics,actual_generations



# ============================================================
# TEST BLOCK
# ============================================================

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

    best_solution, best_metrics = run_genetic_algorithm(
        instance,
        population_size=108,
        generations=200,
        mutation_rate=0.057,
        elite_size=5,
        tournament_size=3,
        stagnation_k=26,
        seed=42,
    )

    print_solution(best_solution)
    print_metrics(best_metrics)
    
