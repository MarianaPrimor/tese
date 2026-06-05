from generate_instance import load_real_instance
from geneticalgorithm import run_genetic_algorithm
from evaluator import print_solution, print_metrics

instance = load_real_instance("../Inputs_Doceleia_small.xlsx")

solution, metrics = run_genetic_algorithm(
    instance,
    population_size=100,
    generations=200,
    mutation_rate=0.10,
    elite_size=5,
    tournament_size=3,
    seed=42,
)

print_solution(solution)
print_metrics(metrics)