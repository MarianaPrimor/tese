from gurobi import solve_with_gurobi


def build_compatibility_test_instance():
    return {
        "n_days": 1,
        "final_lines": ["L1", "L2"],
        "days": ["day_1"],
        "working_days": [None],
        "holidays": [],
        "standard_operators": 10,
        "line_capacity_min": 200,
        "end_of_day_cleaning_time_min": 0,
        "cleaning_operators": 0,
        "available_line_time_min": 200,
        "refs": [
            {
                "id": "ONLY_L2",
                "name": "Product only L2",
                "family": "family_a",
                "cakes_per_box": 1,
                "lead_time_L0_days": 1,
                "can_L1": False,
                "rate_L1_prod": 0,
                "rate_L1_finish": 0,
                "ops_L1_prod": 0,
                "ops_L1_finish": 0,
                "can_L2": True,
                "rate_L2_prod": 60,
                "rate_L2_finish": 60,
                "ops_L2_prod": 3,
                "ops_L2_finish": 0,
            }
        ],
        "families": ["family_a"],
        "setups_matrix": {("family_a", "family_a"): 0},
        "operators": [],
        "competencies": {},
        "demand": [
            {
                "ref_id": "ONLY_L2",
                "master_boxes": 100,
                "delivery_date": 1,
                "priority": "Medium",
            }
        ],
        "machines": {},
        "ref_machine_requirements": {},
        "structure": {},
        "_meta": {},
    }


if __name__ == "__main__":
    solve_with_gurobi(build_compatibility_test_instance())
