import csv
import statistics
import time

from generate_instance import load_real_instance
from geneticalgorithm import run_genetic_algorithm


# Fixed configuration
INSTANCE_PATH = "../Inputs_Doceleia.xlsx"
N_RUNS = 5
SEEDS = [0, 7, 13, 42, 99]
MAX_GENERATIONS = 200


instance = load_real_instance(INSTANCE_PATH)


def run_batch(param_name, param_values, fixed_params):
    """Run several GA repetitions for each value and return summary rows."""
    summary = []

    for value in param_values:
        fitnesses = []
        generations = []
        times = []
        params = dict(fixed_params)
        params[param_name] = value

        print(f"\n[{param_name} = {value}]")

        for i, seed in enumerate(SEEDS):
            start_time = time.time()

            _, metrics, actual_generations = run_genetic_algorithm(
                instance,
                generations=MAX_GENERATIONS,
                seed=seed,
                verbose=False,
                **params,
            )

            elapsed = time.time() - start_time
            fitness = metrics["total_penalty"]

            fitnesses.append(fitness)
            generations.append(actual_generations)
            times.append(elapsed)

            print(
                f"  run {i + 1}/{N_RUNS} | "
                f"fitness={fitness:.0f} | "
                f"gens={actual_generations} | "
                f"time={elapsed:.1f}s"
            )

        row = {
            param_name: value,
            "mean_fitness": round(statistics.mean(fitnesses), 2),
            "std_fitness": round(statistics.stdev(fitnesses), 2),
            "best_fitness": round(min(fitnesses), 2),
            "mean_generations": round(statistics.mean(generations), 1),
            "mean_time_s": round(statistics.mean(times), 2),
        }

        summary.append(row)

        print(
            f"  mean={statistics.mean(fitnesses):.0f} | "
            f"std={statistics.stdev(fitnesses):.0f} | "
            f"best={min(fitnesses):.0f}"
        )

    return summary


def save_summary(path, summary):
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)


#print("\n" + "=" * 50)
#print("EXPERIMENT 1 - POPULATION SIZE")

#summary_pop = run_batch(
   # param_name="population_size",
    #param_values=[50, 100, 150, 200],
    #fixed_params={
      #  "mutation_rate": 0.10,
       # "stagnation_k": 20,
       # "elite_size": 5,
      #  "tournament_size": 3,
   # },
#)

#save_summary("results_population_size.csv", summary_pop)

best_pop = 100
print(f"\nBest population size from experiment 1: {best_pop}")


print("\n" + "=" * 50)
print("EXPERIMENT 2 - MUTATION RATE")

summary_mut = run_batch(
    param_name="mutation_rate",
    param_values=[0.01, 0.03, 0.05, 0.08, 0.10, 0.15],
    fixed_params={
        "population_size": best_pop,
        "stagnation_k": 20,
        "elite_size": 5,
        "tournament_size": 3,
    },
)

save_summary("results_mutation_rate.csv", summary_mut)

best_mut = min(summary_mut, key=lambda x: x["mean_fitness"])["mutation_rate"]
print(f"\nBest mutation rate from experiment 2: {best_mut}")


print("\n" + "=" * 50)
print("EXPERIMENT 3 - STOPPING CRITERION")

summary_stop = run_batch(
    param_name="stagnation_k",
    param_values=[5, 10, 20, 30, 50, MAX_GENERATIONS],
    fixed_params={
        "population_size": best_pop,
        "mutation_rate": best_mut,
        "elite_size": 5,
        "tournament_size": 3,
    },
)

save_summary("results_stagnation_k.csv", summary_stop)

print("\nAll experiments complete. Results saved to CSV files.")
