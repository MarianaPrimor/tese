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
    calculate_operator_usage_by_time,
    get_capacity_tolerance_for_day,
    DELAY_PENALTY,
    POSTPONEMENT_PENALTY,
    ECONOMIC_VALUE_REWARD,
    TIME_BUCKET_MIN,
    compute_max_values,
    normalised_fitness,
    normalised_fitness_breakdown,
)


# GENERATING INITIAL POPULATION


def generate_initial_population(instance, population_size=200, seed=0):
    population = []

    for i in range(population_size):
        solution_seed = seed * 100000 + i
        solution = generate_random_solution(instance, seed=solution_seed)
        solution = enforce_hard_constraints(solution, instance)
        population.append(solution)

    return population



# FITNESS


def fitness(solution, instance, max_values):
    metrics = evaluate_solution(solution, instance)
    return normalised_fitness(metrics, max_values)


def evaluate_with_normalised_fitness(solution, instance, max_values):
    metrics = evaluate_solution(solution, instance)
    raw_total_penalty = metrics["total_penalty"]
    breakdown = normalised_fitness_breakdown(metrics, max_values)
    score = breakdown["total"]

    metrics["raw_total_penalty"] = raw_total_penalty
    metrics["normalised_fitness"] = score
    metrics["normalised_fitness_breakdown"] = breakdown
    metrics["total_penalty"] = score
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


def enforce_hard_constraints(solution, instance):
    refs_by_id = create_refs_by_id(instance)
    repaired_solution = deepcopy(solution)

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
    )
    repaired_solution = repair_operator_constraints(
        repaired_solution,
        instance,
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
):
    candidates = random.sample(population, tournament_size)

    best_candidate = min(
        candidates,
        key=lambda solution: fitness(solution, instance, max_values)
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
    mutation_rate=0.10,
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


def get_postponement_objective_cost(gene, ref):
    economic_value = get_order_economic_value(gene, ref)
    day = gene.get("day")
    due_day = gene.get("delivery_date")
    avoided_delay_cost = 0

    if day is not None and due_day is not None and day > due_day:
        avoided_delay_cost = (day - due_day) * DELAY_PENALTY

    return max(
        1,
        gene.get("master_boxes", 0) * POSTPONEMENT_PENALTY
        + economic_value * ECONOMIC_VALUE_REWARD
        - avoided_delay_cost,
    )


def repair_capacity_constraints(solution, instance):
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
                )
                candidates.append((
                    postponement_cost / freed_time,
                    index,
                ))

            _, selected_index = min(candidates)
            selected_gene = genes.pop(selected_index)
            postpone_gene(selected_gene)

    return repaired_solution


def repair_operator_constraints(solution, instance):
    refs_by_id = create_refs_by_id(instance)
    repaired_solution = deepcopy(solution)
    max_iterations = len(repaired_solution)

    for _ in range(max_iterations):
        operations = simulate_time_schedule(repaired_solution, instance)
        _, excess_by_time = calculate_operator_usage_by_time(
            operations,
            instance,
        )
        violating_slots = {
            key: excess
            for key, excess in excess_by_time.items()
            if excess > 0
        }

        if not violating_slots:
            break

        relief_by_order = {}

        for operation in operations:
            order_id = operation.get("order_id")

            if order_id is None or operation["end"] <= operation["start"]:
                continue

            start_bucket = int(operation["start"] // TIME_BUCKET_MIN)
            end_bucket = int((operation["end"] - 1) // TIME_BUCKET_MIN)

            for bucket in range(start_bucket, end_bucket + 1):
                key = (operation["day"], bucket)

                if key not in violating_slots:
                    continue

                relief_by_order[order_id] = (
                    relief_by_order.get(order_id, 0)
                    + min(
                        operation.get("operators", 0),
                        violating_slots[key],
                    )
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
                cost = get_postponement_objective_cost(gene, ref)

            candidates.append((cost / relief, order_id, gene))

        if not candidates:
            break

        _, _, selected_gene = min(
            candidates,
            key=lambda candidate: (candidate[0], candidate[1]),
        )
        postpone_gene(selected_gene)

    return repaired_solution

# ============================================================
# GENETIC ALGORITHM
# ============================================================

def run_genetic_algorithm(
    instance,
    population_size=100,
    generations=200,
    mutation_rate=0.10,
    elite_size=5,
    tournament_size=3,
    stagnation_k=20,
    seed=0,
    verbose= True,
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
        seed=seed
    )

    if verbose:
        print("[GA] Evaluating initial population...", flush=True)

    random.seed(seed)
    population = sorted(
        population,
        key=lambda solution: fitness(solution, instance, max_values)
    )
    initial_best_solution = deepcopy(population[0])
    initial_best_metrics = evaluate_with_normalised_fitness(
        initial_best_solution,
        instance,
        max_values,
    )

    best_solution = deepcopy(initial_best_solution)
    best_metrics = initial_best_metrics
    generations_without_improvement = 0
    actual_generations = 0
    
    for generation in range(1, generations + 1):
        actual_generations = generation
        population = sorted(
            population,
            key=lambda solution: fitness(solution, instance, max_values)
        )

        current_best = population[0]
        current_metrics = evaluate_with_normalised_fitness(
            current_best,
            instance,
            max_values,
        )
        if current_metrics["total_penalty"] < best_metrics["total_penalty"]:
            best_solution = deepcopy(current_best)
            best_metrics = current_metrics
            generations_without_improvement = 0  # reset
        else:
            generations_without_improvement += 1  # increment

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
            or current_metrics["total_penalty"] < best_metrics["total_penalty"]
        ):
            best_solution = deepcopy(current_best)
            best_metrics = current_metrics

        if verbose:
            print(
                f"Generation {generation:03d} | "
                f"best fitness: {best_metrics['total_penalty']:.6f} | "
                f"current best: {current_metrics['total_penalty']:.6f}",
                flush=True,
            )


        new_population = [
            enforce_hard_constraints(solution, instance)
            for solution in population[:elite_size]
        ]

        while len(new_population) < population_size:
            parent_1 = tournament_selection(
                population,
                instance,
                max_values,
                tournament_size=tournament_size
            )

            parent_2 = tournament_selection(
                population,
                instance,
                max_values,
                tournament_size=tournament_size
            )

            child = crossover(parent_1, parent_2)
            child = mutate(
                child,
                instance,
                mutation_rate=mutation_rate,
            )

            child = enforce_hard_constraints(child, instance)

            new_population.append(child)

        population = new_population

    best_solution = enforce_hard_constraints(best_solution, instance)
    best_metrics = evaluate_with_normalised_fitness(
        best_solution,
        instance,
        max_values,
    )
    best_metrics["computation_time_sec"] = time.perf_counter() - start_time

    initial_penalty = initial_best_metrics["total_penalty"]
    final_penalty = best_metrics["total_penalty"]

    if initial_penalty != 0:
        improvement = (
            (initial_penalty - final_penalty)
            / abs(initial_penalty)
        ) * 100
    else: 
        improvement = 0.0

    print("\n=== GENETIC ALGORITHM SUMMARY ===")
    print(f"Initial best fitness: {initial_penalty:.6f}")
    print(f"Final best fitness: {final_penalty:.6f}")
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
        "Operator utilisation contribution: "
        f"{breakdown['operator_utilisation']:+.6f}"
    )
    print(f"Normalised fitness Z: {breakdown['total']:+.6f}")
    print(f"Maximum postponed volume: {max_values['postponed_volume']:.2f}")
    print(f"Maximum delay days: {max_values['delay_days']:.2f}")
    print(f"Maximum setup time: {max_values['setup_time']:.2f}")
    print(f"Maximum economic value: {max_values['economic_value']:.2f}")
    print(f"Maximum operator minutes: {max_values['operator_minutes']:.2f}")

    return best_solution, best_metrics,actual_generations



# ============================================================
# TEST BLOCK
# ============================================================

if __name__ == "__main__":
    instance = load_real_instance("../Inputs_EmpresaX.xlsx")

    best_solution, best_metrics = run_genetic_algorithm(
        instance,
        population_size=200,
        generations=200,
        mutation_rate=0.10,
        elite_size=5,
        tournament_size=3,
        seed=42,
    )

    print_solution(best_solution)
    print_metrics(best_metrics)
    
