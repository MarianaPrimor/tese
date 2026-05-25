from copy import deepcopy
import random

from generate_instance import load_real_instance
from evaluator import (
    generate_random_solution,
    evaluate_solution,
    print_solution,
    print_metrics,
    create_refs_by_id,
    valid_lines_for_ref,
    get_setup,
)


# GENERATING INITIAL POPULATION


def generate_initial_population(instance, population_size=100):
    population = []

    for seed in range(population_size):
        solution = generate_random_solution(instance, seed=seed)
        solution = repair_capacity(solution, instance)
        population.append(solution)

    return population



# FITNESS


def fitness(solution, instance):
    metrics = evaluate_solution(solution, instance)
    return metrics["total_penalty"]


# ============================================================
# SELECTION
# ============================================================

def tournament_selection(population, instance, tournament_size=3):
    candidates = random.sample(population, tournament_size)

    best_candidate = min(
        candidates,
        key=lambda solution: fitness(solution, instance)
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
    assignment_mutation_rate=0.05,
    postponement_mutation_rate=0.30,
):
    refs_by_id = create_refs_by_id(instance)
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

    # Assignment mutation: changes day/line/postponement while keeping feasibility
    for gene in mutated_solution:
        if random.random() < assignment_mutation_rate:
            ref_id = str(gene["ref_id"]).strip()

            if ref_id not in refs_by_id:
                gene["line"] = None
                gene["day"] = None
                gene["postponed"] = True
                continue

            ref = refs_by_id[ref_id]
            valid_lines = valid_lines_for_ref(ref)

            if random.random() < postponement_mutation_rate:
                gene["postponed"] = not gene.get("postponed", False)

            if gene.get("postponed") or not valid_lines:
                gene["line"] = None
                gene["day"] = None
                gene["postponed"] = True
            else:
                gene["line"] = random.choice(valid_lines)
                gene["day"] = random.randint(1, instance["n_days"])

    return mutated_solution


def get_production_time_for_repair(ref, line, master_boxes):
    if line == "L1":
        rate = ref["rate_L1_prod"]
    elif line == "L2":
        rate = ref["rate_L2_prod"]
    else:
        return None

    if rate is None or rate == 0:
        return None

    n_cakes = master_boxes * ref["cakes_per_box"]
    return (n_cakes / rate) * 60


def repair_capacity(solution, instance):
    repaired_solution = deepcopy(solution)

    refs_by_id = create_refs_by_id(instance)
    available_capacity = instance["available_line_time_min"]

    groups = {}

    for gene in repaired_solution:
        if gene.get("postponed"):
            continue

        key = (gene.get("day"), gene.get("line"))
        groups.setdefault(key, []).append(gene)

    for key, genes in groups.items():
        day, line = key

        if day is None or line is None:
            continue

        total_time = 0
        feasible_genes = []

        for gene in genes:
            ref_id = str(gene["ref_id"]).strip()

            if ref_id not in refs_by_id:
                continue

            ref = refs_by_id[ref_id]
            production_time = get_production_time_for_repair(
                ref,
                line,
                gene["master_boxes"]
            )

            if production_time is None:
                continue

            total_time += production_time
            feasible_genes.append((gene, production_time))

        if total_time <= available_capacity:
            continue

        feasible_genes = sorted(
            feasible_genes,
            key=lambda x: (
                x[0].get("delivery_date", instance["n_days"]),
                -x[1]
            ),
            reverse=True
        )

        while total_time > available_capacity and feasible_genes:
            gene, production_time = feasible_genes.pop(0)

            gene["postponed"] = True
            gene["day"] = None
            gene["line"] = None

            total_time -= production_time

    return repaired_solution

# ============================================================
# GENETIC ALGORITHM
# ============================================================

def run_genetic_algorithm(
    instance,
    population_size=100,
    generations=100,
    mutation_rate=0.10,
    elite_size=5,
    tournament_size=3,
    seed=42,
    max_generations_without_improvement=10,
):
    random.seed(seed)

    population = generate_initial_population(instance, population_size)
    population = sorted(
        population,
        key=lambda solution: fitness(solution, instance)
    )
    initial_best_solution = deepcopy(population[0])
    initial_best_metrics = evaluate_solution(initial_best_solution, instance)

    best_solution = deepcopy(initial_best_solution)
    best_metrics = initial_best_metrics
    
    generations_without_improvement = 0
    for generation in range(1, generations + 1):
        population = sorted(
            population,
            key=lambda solution: fitness(solution, instance)
        )

        current_best = population[0]
        current_metrics = evaluate_solution(current_best, instance)

        if (
            best_metrics is None
            or current_metrics["total_penalty"] < best_metrics["total_penalty"]
        ):
            best_solution = deepcopy(current_best)
            best_metrics = current_metrics
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        print(
            f"Generation {generation:03d} | "
            f"best penalty: {best_metrics['total_penalty']:.2f} | "
            f"current best: {current_metrics['total_penalty']:.2f}"
        )

        if generations_without_improvement >= max_generations_without_improvement:
            print(
                f"Stopping early: no improvement for "
                f"{max_generations_without_improvement} generations."
            )
            break   

        new_population = [
            deepcopy(solution)
            for solution in population[:elite_size]
        ]

        while len(new_population) < population_size:
            parent_1 = tournament_selection(
                population,
                instance,
                tournament_size=tournament_size
            )

            parent_2 = tournament_selection(
                population,
                instance,
                tournament_size=tournament_size
            )

            child = crossover(parent_1, parent_2)
            child = mutate(
                child,
                instance,
                mutation_rate=mutation_rate,
                assignment_mutation_rate=0.05
            )

            child = repair_capacity(child, instance)

            new_population.append(child)

        population = new_population

    initial_penalty = initial_best_metrics["total_penalty"]
    final_penalty = best_metrics["total_penalty"]

    if initial_penalty > 0:
        improvement = ((initial_penalty - final_penalty) / initial_penalty )* 100
    else: 
        improvement = 0.0

    print("\n=== GENETIC ALGORITHM SUMMARY ===")
    print(f"Initial best penalty: {initial_penalty:.2f}")
    print(f"Final best penalty: {final_penalty:.2f}")
    print(f"Improvement: {improvement:.2f}%")
    print(f"Initial delay: {initial_best_metrics['delay_days_total']} days")
    print(f"Final delay: {best_metrics['delay_days_total']} days")
    print(f"Initial capacity excess: {initial_best_metrics['total_capacity_excess']:.2f} min")
    print(f"Final capacity excess: {best_metrics['total_capacity_excess']:.2f} min")
    print(f"Initial operator excess: {initial_best_metrics['total_operator_excess']:.2f}")
    print(f"Final operator excess: {best_metrics['total_operator_excess']:.2f}")
    print(f"Initial setup time: {initial_best_metrics['setup_total_min']:.2f} min")
    print(f"Final setup time: {best_metrics['setup_total_min']:.2f} min")

    return best_solution, best_metrics



# ============================================================
# TEST BLOCK
# ============================================================

if __name__ == "__main__":
    instance = load_real_instance("../Inputs_Doceleia.xlsx")

    best_solution, best_metrics = run_genetic_algorithm(
        instance,
        population_size=100,
        generations=100,
        mutation_rate=0.10,
        elite_size=5,
        tournament_size=3,
        seed=42,
    )

    print_solution(best_solution)
    print_metrics(best_metrics)
    
