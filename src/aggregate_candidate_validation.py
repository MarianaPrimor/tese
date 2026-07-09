from pathlib import Path

import pandas as pd

from evaluate_optuna_candidates import summarise


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "candidate_parameter_validation"


def main():
    files = sorted(OUTPUT_DIR.glob("raw_*.csv"))
    if not files:
        raise SystemExit("No raw candidate validation CSV files found.")

    raw_df = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    raw_df = raw_df.sort_values(["candidate", "instance", "seed"])
    by_instance = summarise(raw_df, ["candidate", "instance"])
    global_summary = summarise(raw_df, ["candidate"])
    global_summary = global_summary.sort_values("Z_mean")
    global_summary["rank_by_mean"] = range(1, len(global_summary) + 1)
    global_summary["worst_instance_seed_Z"] = (
        raw_df.groupby("candidate")["Z"].max().reindex(global_summary["candidate"]).values
    )
    global_summary = global_summary.sort_values("worst_instance_seed_Z")
    global_summary["rank_by_minimax"] = range(1, len(global_summary) + 1)
    global_summary = global_summary.sort_values("rank_by_mean")

    raw_df.to_csv(OUTPUT_DIR / "candidate_validation_raw.csv", index=False)
    by_instance.to_csv(OUTPUT_DIR / "candidate_validation_by_instance.csv", index=False)
    global_summary.to_csv(OUTPUT_DIR / "candidate_validation_global_summary.csv", index=False)

    print(global_summary[[
        "candidate",
        "population_size",
        "mutation_rate",
        "stagnation_k",
        "Z_mean",
        "Z_std",
        "Z_min",
        "Z_max",
        "worst_instance_seed_Z",
        "rank_by_mean",
        "rank_by_minimax",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
