"""
Benchmark Dataset Loader and Validator (W6)

Enforces strict separation between:
1. Development fixtures (sample_type="development")
2. Synthetic regression fixtures (sample_type="synthetic")
3. Research benchmark subsets (sample_type="research")

Guarantees that development and synthetic fixtures are never accidentally
included in research evaluation reports.
"""

import json
import logging
import pathlib
from typing import List, Optional

from evaluation.schemas import (
    BenchmarkDataset,
    BenchmarkSample,
    ReferenceHighlight,
    SampleType,
)

logger = logging.getLogger("clipsense.evaluation.dataset")


def load_benchmark_dataset(
    dataset_path: str,
    required_sample_type: Optional[SampleType] = None,
) -> BenchmarkDataset:
    """
    Loads and validates a benchmark dataset from a JSON file.
    
    If required_sample_type == "research", strictly prevents any sample with
    sample_type == "development" or sample_type == "synthetic" from being loaded.
    """
    path = pathlib.Path(dataset_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Benchmark dataset manifest not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset = BenchmarkDataset(**data)

    # Validate dataset type consistency
    if required_sample_type and dataset.dataset_type != required_sample_type:
        raise ValueError(
            f"Dataset type mismatch: requested '{required_sample_type}', "
            f"but dataset manifest declares '{dataset.dataset_type}'."
        )

    # Validate each sample's sample_type
    for sample in dataset.samples:
        if required_sample_type == "research" and sample.sample_type != "research":
            raise ValueError(
                f"Contamination error: Sample '{sample.sample_id}' has sample_type='{sample.sample_type}', "
                f"which is strictly prohibited in research benchmark evaluations."
            )
        if sample.sample_type != dataset.dataset_type:
            raise ValueError(
                f"Inconsistent sample type: Sample '{sample.sample_id}' declares type '{sample.sample_type}', "
                f"differing from dataset-level type '{dataset.dataset_type}'."
            )

    logger.info(
        f"Loaded benchmark dataset '{dataset.dataset_name}' with {len(dataset.samples)} "
        f"samples (type: {dataset.dataset_type})."
    )
    return dataset


def create_synthetic_regression_fixture() -> BenchmarkDataset:
    """
    Creates an in-memory synthetic fixture strictly for CI and unit tests.
    Never used or reported as a research benchmark.
    """
    return BenchmarkDataset(
        dataset_name="synthetic_regression_testbed",
        dataset_type="synthetic",
        description="Deterministic synthetic scenarios for CI/unit testing only.",
        samples=[
            BenchmarkSample(
                sample_id="synthetic_case_1",
                sample_type="synthetic",
                duration_sec=60.0,
                run_dir="mock_synthetic_run_1",
                reference_spans=[
                    ReferenceHighlight(
                        span_id="ref_synth_1",
                        start_time=10.0,
                        end_time=30.0,
                        annotation_source="synthetic_testbed",
                        description="Synthetic ground truth consensus span 10s-30s",
                    )
                ],
                metadata={"category": "synthetic_unit_test"},
            ),
            BenchmarkSample(
                sample_id="synthetic_case_2",
                sample_type="synthetic",
                duration_sec=90.0,
                run_dir="mock_synthetic_run_2",
                reference_spans=[
                    ReferenceHighlight(
                        span_id="ref_synth_2a",
                        start_time=5.0,
                        end_time=25.0,
                        annotation_source="synthetic_testbed",
                        description="Synthetic consensus span 5s-25s",
                    ),
                    ReferenceHighlight(
                        span_id="ref_synth_2b",
                        start_time=45.0,
                        end_time=70.0,
                        annotation_source="synthetic_testbed",
                        description="Synthetic consensus span 45s-70s",
                    ),
                ],
                metadata={"category": "synthetic_unit_test"},
            ),
        ],
    )
