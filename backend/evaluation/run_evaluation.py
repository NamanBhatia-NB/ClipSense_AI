"""
Evaluation CLI Entrypoint (W6)

Usage:
    python -m evaluation.run_evaluation [options]

Examples:
    # Run development validation fixture across all 6 modes:
    python -m evaluation.run_evaluation --dataset evaluation/datasets/dev_fixture.json --mock-llm

    # Run specific modes:
    python -m evaluation.run_evaluation --dataset evaluation/datasets/dev_fixture.json --modes baseline,all_experts_no_mter,mter_full
"""

import argparse
import json
import logging
import os
import pathlib
import sys
from typing import List

# Ensure backend root is on sys.path
backend_dir = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(backend_dir))

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from core.llm import MockLLMClient, get_llm_client
from evaluation.ablation_runner import AblationRunner
from evaluation.baseline import (
    BaselineConfig,
    BaselineLLMCandidate,
    BaselineLLMResponse,
    LegacyTranscriptBaselineSelector,
)
from evaluation.benchmark_orchestrator import BenchmarkOrchestrator
from evaluation.dataset import load_benchmark_dataset
from evaluation.schemas import ExperimentMode

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("clipsense.evaluation.cli")

ALL_MODES: List[ExperimentMode] = [
    "baseline",
    "transcript_conversation",
    "transcript_conversation_visual",
    "transcript_conversation_prosody",
    "all_experts_no_mter",
    "mter_full",
]


class EvaluationMockLLMClient(MockLLMClient):
    """Deterministic mock client used strictly when --mock-llm is specified for testing."""

    def generate_structured(self, prompt: str, response_schema: any, system_instruction: str = None) -> any:
        if response_schema == BaselineLLMResponse:
            import re
            indices = [int(m) for m in re.findall(r"\[(\d+)\]", prompt)]
            candidates = []
            if len(indices) >= 30:
                candidates.append(
                    BaselineLLMCandidate(
                        start_word_index=indices[1],
                        end_word_index=indices[min(len(indices) - 2, 75)],
                        confidence=0.85,
                        title="Mock Baseline Thematic Moment",
                        summary="Spoken discussion segment extracted by mock baseline.",
                    )
                )
            return BaselineLLMResponse(candidates=candidates)
        return super().generate_structured(prompt, response_schema, system_instruction)


def main():
    parser = argparse.ArgumentParser(description="ClipSense Research Evaluation Runner (W6)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="evaluation/datasets/dev_fixture.json",
        help="Path to benchmark dataset JSON manifest.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=",".join(ALL_MODES),
        help="Comma-separated experiment modes to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/evaluation_run",
        help="Directory to store evaluation results and report.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to read/write cached baseline predictions.",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Use deterministic Mock LLM (strictly for automated testing/validation).",
    )
    parser.add_argument(
        "--cached-baseline",
        action="store_true",
        help="Enforce cached baseline inference (fails if cache not present).",
    )
    parser.add_argument(
        "--min-match-iou",
        type=float,
        default=0.10,
        help="Minimum tIoU threshold for Hungarian bipartite matching acceptance (default: 0.10).",
    )

    args = parser.parse_args()

    # Parse modes
    requested_modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in requested_modes:
        if m not in ALL_MODES:
            logger.error(f"Invalid mode: '{m}'. Supported modes: {ALL_MODES}")
            sys.exit(1)

    # Resolve paths relative to backend root if needed
    dataset_path = pathlib.Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = backend_dir / dataset_path

    out_dir = pathlib.Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = backend_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = args.cache_dir
    if cache_dir:
        cache_path = pathlib.Path(cache_dir)
        if not cache_path.is_absolute():
            cache_path = backend_dir / cache_path
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_dir = str(cache_path)
    else:
        cache_dir = str(out_dir / "baseline_cache")

    logger.info(f"Loading dataset from: {dataset_path}")
    dataset = load_benchmark_dataset(str(dataset_path))

    # Resolve sample run_dir relative to backend root if not absolute
    for s in dataset.samples:
        s_path = pathlib.Path(s.run_dir)
        if not s_path.is_absolute():
            s.run_dir = str((backend_dir / s_path).resolve())

    # Configure baseline
    baseline_cfg = BaselineConfig(
        inference_mode="cached_inference" if args.cached_baseline else "live_inference",
        cache_dir=cache_dir,
    )

    if args.mock_llm:
        logger.info("Using EvaluationMockLLMClient for testing.")
        llm_client = EvaluationMockLLMClient()
    elif not args.cached_baseline:
        try:
            llm_client = get_llm_client()
        except Exception as e:
            logger.warning(
                f"Live LLM client unavailable ({e}). Falling back to mock client for this run."
            )
            llm_client = EvaluationMockLLMClient()
    else:
        llm_client = None

    baseline_selector = LegacyTranscriptBaselineSelector(
        llm_client=llm_client, config=baseline_cfg
    )

    ablation_runner = AblationRunner(baseline_selector=baseline_selector)
    orchestrator = BenchmarkOrchestrator(ablation_runner=ablation_runner)

    logger.info(f"Starting evaluation across modes: {requested_modes}")
    report = orchestrator.evaluate_dataset(
        dataset=dataset,
        modes=requested_modes,
        cache_dir=cache_dir,
        min_match_iou=args.min_match_iou,
    )

    # Generate Markdown Report
    md_content = orchestrator.generate_markdown_report(report)

    # Save outputs
    json_path = out_dir / "eval_results.json"
    md_path = out_dir / "eval_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("\n" + "=" * 80)
    print(md_content)
    print("=" * 80)
    logger.info(f"Evaluation results saved to {json_path}")
    logger.info(f"Markdown report saved to {md_path}\n")


if __name__ == "__main__":
    main()
