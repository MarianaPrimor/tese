import argparse
from pathlib import Path

import pandas as pd

from evaluate_optuna_candidates import run_candidate_seed


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "candidate_parameter_validation"


def split_dates(value):
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Run one candidate/instance/seed validation.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--source-month", required=True)
    parser.add_argument("--population-size", type=int, required=True)
    parser.add_argument("--mutation-rate", type=float, required=True)
    parser.add_argument("--stagnation-k", type=int, required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--instance-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--operators", type=int, default=20)
    parser.add_argument("--non-working-dates", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidate = {
        "candidate": args.candidate,
        "source_month": args.source_month,
        "source_trial": None,
        "source_mean_fitness": None,
        "population_size": args.population_size,
        "mutation_rate": args.mutation_rate,
        "stagnation_k": args.stagnation_k,
    }
    row = run_candidate_seed(
        candidate,
        args.instance,
        args.instance_file.resolve(),
        args.operators,
        split_dates(args.non_working_dates),
        args.seed,
    )
    output = OUTPUT_DIR / f"raw_{args.candidate}_{args.instance}_seed{args.seed}.csv"
    pd.DataFrame([row]).to_csv(output, index=False)
    print(f"Saved {output}")
    print(f"{args.candidate} on {args.instance} seed {args.seed}: Z={row['Z']:.8f}")


if __name__ == "__main__":
    main()
