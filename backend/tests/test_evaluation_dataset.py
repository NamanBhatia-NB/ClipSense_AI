"""
Unit Tests for Benchmark Dataset Loader and Sample Type Guards (W6)
"""

import json
import pathlib
import tempfile
import unittest

from evaluation.dataset import (
    create_synthetic_regression_fixture,
    load_benchmark_dataset,
)


class TestEvaluationDataset(unittest.TestCase):

    def setUp(self):
        self.backend_dir = pathlib.Path(__file__).parent.parent.resolve()
        self.dev_fixture_path = self.backend_dir / "evaluation" / "datasets" / "dev_fixture.json"
        self.synthetic_suite_path = self.backend_dir / "evaluation" / "datasets" / "synthetic_suite.json"

    def test_load_dev_fixture(self):
        dataset = load_benchmark_dataset(str(self.dev_fixture_path))
        self.assertEqual(dataset.dataset_type, "development")
        self.assertEqual(len(dataset.samples), 1)
        self.assertEqual(dataset.samples[0].sample_type, "development")
        self.assertEqual(
            dataset.samples[0].reference_spans[0].annotation_source, "manual_dev_debug"
        )

    def test_load_synthetic_fixture(self):
        dataset = load_benchmark_dataset(str(self.synthetic_suite_path))
        self.assertEqual(dataset.dataset_type, "synthetic")
        self.assertEqual(dataset.samples[0].sample_type, "synthetic")

    def test_in_memory_synthetic_fixture(self):
        dataset = create_synthetic_regression_fixture()
        self.assertEqual(dataset.dataset_type, "synthetic")
        self.assertEqual(len(dataset.samples), 2)

    def test_research_safety_guard_blocks_contamination(self):
        # Attempting to load dev fixture under required_sample_type="research" must raise ValueError
        with self.assertRaises(ValueError) as ctx:
            load_benchmark_dataset(
                str(self.dev_fixture_path), required_sample_type="research"
            )
        self.assertIn("Dataset type mismatch", str(ctx.exception))

    def test_sample_level_contamination_guard(self):
        # Create a malformed dataset declaring 'research' but containing a 'development' sample
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "dataset_name": "contaminated_dataset",
                    "dataset_type": "research",
                    "description": "Malformed dataset with contaminated sample",
                    "samples": [
                        {
                            "sample_id": "bad_sample",
                            "sample_type": "development",
                            "duration_sec": 30.0,
                            "run_dir": "runs/test",
                            "reference_spans": [],
                        }
                    ],
                },
                f,
            )
            tmp_name = f.name

        try:
            with self.assertRaises(ValueError) as ctx:
                load_benchmark_dataset(tmp_name, required_sample_type="research")
            self.assertIn("Contamination error", str(ctx.exception))
        finally:
            pathlib.Path(tmp_name).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
