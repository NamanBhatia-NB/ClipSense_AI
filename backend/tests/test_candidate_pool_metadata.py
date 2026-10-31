"""
ClipSense AI — Candidate Pool Selection Budget Metadata Regression Tests

Verifies that:
1. The helper function `get_system_selection_budget_mode` returns:
   - 'full_returned_candidate_pool' for System A, System B, and Uniform Control.
   - 'top_k_capped_20' for System C and System D.
2. The canonical Rhapsody timestamp predictions CSV (`predictions_timestamp_only.csv`)
   labels every row with the appropriate candidate-pool selection mode.
"""

import csv
import pathlib
import unittest

from evaluation.run_timestamp_evaluation import get_system_selection_budget_mode


class TestCandidatePoolSelectionBudgetMetadata(unittest.TestCase):
    def test_system_selection_budget_mode_mapping(self):
        """Verify unit mapping logic for each system identifier."""
        # System A: full returned candidate pool
        self.assertEqual(
            get_system_selection_budget_mode("system_a_transcript_only"),
            "full_returned_candidate_pool",
        )
        self.assertEqual(
            get_system_selection_budget_mode("SYS_A_SEMANTIC"),
            "full_returned_candidate_pool",
        )

        # System B: full returned candidate pool
        self.assertEqual(
            get_system_selection_budget_mode("system_b_simple_aggregation"),
            "full_returned_candidate_pool",
        )
        self.assertEqual(
            get_system_selection_budget_mode("SYS_B_AGGR"),
            "full_returned_candidate_pool",
        )

        # Control: full returned candidate pool
        self.assertEqual(
            get_system_selection_budget_mode("system_a_uniform_window"),
            "full_returned_candidate_pool",
        )
        self.assertEqual(
            get_system_selection_budget_mode("SYS_A_UNIFORM_CTRL"),
            "full_returned_candidate_pool",
        )

        # System C: top-k capped at 20
        self.assertEqual(
            get_system_selection_budget_mode("system_c_mter"),
            "top_k_capped_20",
        )
        self.assertEqual(
            get_system_selection_budget_mode("SYS_C_MTER"),
            "top_k_capped_20",
        )

        # System D: top-k capped at 20
        self.assertEqual(
            get_system_selection_budget_mode("system_d_mter_refined"),
            "top_k_capped_20",
        )
        self.assertEqual(
            get_system_selection_budget_mode("SYS_D_REFINED"),
            "top_k_capped_20",
        )

    def test_canonical_predictions_csv_selection_budget_metadata(self):
        """Verify that the canonical predictions CSV contains valid, accurate selection_budget_mode values."""
        repo_root = pathlib.Path(__file__).resolve().parent.parent
        csv_path = repo_root / "evaluation" / "rhapsody_sub3600_eval_output_final" / "predictions_timestamp_only.csv"
        
        self.assertTrue(csv_path.exists(), f"Predictions CSV not found at {csv_path}")

        counts_by_system = {}
        modes_by_system = {}

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.assertIn("selection_budget_mode", reader.fieldnames)
            for row in reader:
                s_name = row["system_name"]
                mode = row["selection_budget_mode"]
                counts_by_system[s_name] = counts_by_system.get(s_name, 0) + 1
                if s_name not in modes_by_system:
                    modes_by_system[s_name] = set()
                modes_by_system[s_name].add(mode)

        expected_modes = {
            "system_a_transcript_only": {"full_returned_candidate_pool"},
            "system_b_simple_aggregation": {"full_returned_candidate_pool"},
            "system_a_uniform_window": {"full_returned_candidate_pool"},
            "system_c_mter": {"top_k_capped_20"},
            "system_d_mter_refined": {"top_k_capped_20"},
        }

        self.assertEqual(set(counts_by_system.keys()), set(expected_modes.keys()))
        for s_name, exp_mode_set in expected_modes.items():
            self.assertEqual(
                modes_by_system[s_name],
                exp_mode_set,
                f"Unexpected selection_budget_mode for {s_name}: {modes_by_system[s_name]} vs {exp_mode_set}",
            )

        # Confirm non-zero expected counts
        self.assertEqual(counts_by_system["system_a_transcript_only"], 195)
        self.assertEqual(counts_by_system["system_b_simple_aggregation"], 295)
        self.assertEqual(counts_by_system["system_a_uniform_window"], 592)
        self.assertEqual(counts_by_system["system_c_mter"], 100)
        self.assertEqual(counts_by_system["system_d_mter_refined"], 100)


if __name__ == "__main__":
    unittest.main()
