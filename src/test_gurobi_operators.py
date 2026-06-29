from gurobi import solve_with_gurobi


def build_operator_test_instance():
    return {
        "n_days": 1,
        "final_lines": ["L1", "L2"],
        "days": ["day_1"],
        "working_days": [None],
        "holidays": [],
        "standard_operators": 5,
        "line_capacity_min": 1000,
        "end_of_day_cleaning_time_min": 0,
        "cleaning_operators": 0,
        "available_line_time_min": 1000,
        "refs": [
            {
                "id": "A_L1",
                "name": "Product A L1",
                "family": "family_a",
                "cakes_per_box": 1,
                "lead_time_L0_days": 1,
                "can_L1": True,
                "rate_L1_prod": 60,
                "rate_L1_finish": 60,
                "ops_L1_prod": 3,
                "ops_L1_finish": 0,
                "can_L2": False,
                "rate_L2_prod": 0,
                "rate_L2_finish": 0,
                "ops_L2_prod": 0,
                "ops_L2_finish": 0,
            },
            {
                "id": "B_L2",
                "name": "Product B L2",
                "family": "family_b",
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
                "ops_L2_prod": 4,
                "ops_L2_finish": 0,
            },
        ],
        "families": ["family_a", "family_b"],
        "setups_matrix": {
            ("family_a", "family_a"): 0,
            ("family_b", "family_b"): 0,
            ("family_a", "family_b"): 0,
            ("family_b", "family_a"): 0,
        },
        "operators": [],
        "competencies": {},
        "demand": [
            {
                "ref_id": "A_L1",
                "master_boxes": 100,
                "delivery_date": 1,
                "priority": "Medium",
            },
            {
                "ref_id": "B_L2",
                "master_boxes": 100,
                "delivery_date": 1,
                "priority": "Medium",
            },
        ],
        "machines": {},
        "ref_machine_requirements": {},
        "structure": {},
        "_meta": {},
    }


if __name__ == "__main__":
    solve_with_gurobi(build_operator_test_instance())
