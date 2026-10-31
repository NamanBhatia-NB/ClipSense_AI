"""
ClipSense AI — Rank Consistency and Human Evaluation Pilot Regression Tests
Deliverable 5 Verification Suite

Regression tests covering:
1. Top-K candidate selection invariance to arbitrary CSV row shuffling.
2. Complete candidate selection isolation from reference-derived ground-truth fields.
3. Accurate reporting of missing candidates when a system has fewer than K candidates.
4. System C and System D base candidate interval identity (refinement ablation).
5. UI rank, filename, and manifest rank consistency.
6. Manifest accuracy for rendered start/end, original timestamps, score, refinement status, and fallback reasons.
7. Verification that human rating form fields are completely blank without fabricated ratings.
"""

import csv
import json
import os
import random
import shutil
import tempfile
import unittest
from pathlib import Path

from core.schemas import (
    BoundaryRefinementMetadata,
    ClipExportResult,
    ClipsManifest,
    MTERCandidate,
    CandidateSupportMetrics,
    SemanticVerificationCriteria,
)
from evaluation.human_eval_export import (
    export_top1_pilot_package,
    extract_candidate_ranking_score,
    select_top_k_candidates_for_system,
)
from pipeline.render.video_renderer import (
    VideoRenderer,
    canonical_candidate_sort_key,
)


class TestRankAndPilotConsistency(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_top_k_selection_invariant_to_row_order(self):
        """Verify that top-K selection is strictly invariant to CSV row shuffling."""
        sample_rows = [
            {"span_id": "span_A", "pred_start": 10.0, "pred_end": 25.0, "pred_duration": 15.0, "inference_score": 0.42},
            {"span_id": "span_B", "pred_start": 50.0, "pred_end": 80.0, "pred_duration": 30.0, "inference_score": 0.88},
            {"span_id": "span_C", "pred_start": 120.0, "pred_end": 145.0, "pred_duration": 25.0, "inference_score": 0.65},
            {"span_id": "span_D", "pred_start": 200.0, "pred_end": 235.0, "pred_duration": 35.0, "inference_score": 0.72},
            {"span_id": "span_E", "pred_start": 300.0, "pred_end": 330.0, "pred_duration": 30.0, "inference_score": 0.88},
        ]

        # Top-2 selected from original order
        baseline_top2 = select_top_k_candidates_for_system(sample_rows, "system_b_simple_aggregation", k=2)
        baseline_ids = [c["span_id"] for c in baseline_top2]

        # Test across 10 random permutations of row order
        rng = random.Random(1337)
        for i in range(10):
            shuffled = list(sample_rows)
            rng.shuffle(shuffled)
            shuffled_top2 = select_top_k_candidates_for_system(shuffled, "system_b_simple_aggregation", k=2)
            shuffled_ids = [c["span_id"] for c in shuffled_top2]
            self.assertEqual(
                shuffled_ids,
                baseline_ids,
                f"Top-2 selection changed on shuffle iteration {i}: {shuffled_ids} vs {baseline_ids}",
            )

    def test_selection_does_not_access_reference_fields(self):
        """Verify that selection is invariant to adversarial ground-truth/reference fields."""
        cands = [
            {
                "span_id": "cand_high_score",
                "pred_start": 100.0,
                "pred_end": 130.0,
                "pred_duration": 30.0,
                "inference_score": 0.95,
                "matched": False,
                "benchmark_start": None,
                "benchmark_end": None,
                "start_error_sec": 999.0,
                "end_error_sec": 999.0,
                "temporal_iou": 0.0,
            },
            {
                "span_id": "cand_low_score_high_ref",
                "pred_start": 200.0,
                "pred_end": 230.0,
                "pred_duration": 30.0,
                "inference_score": 0.10,
                "matched": True,
                "benchmark_start": 200.0,
                "benchmark_end": 230.0,
                "start_error_sec": 0.0,
                "end_error_sec": 0.0,
                "temporal_iou": 1.0,
            },
        ]

        # In System C, high inference score must be chosen despite zero IoU / unmatched
        top_c = select_top_k_candidates_for_system(cands, "system_c_mter", k=1)
        self.assertEqual(top_c[0]["span_id"], "cand_high_score")

        # Strip all reference columns completely
        clean_cands = [
            {"span_id": c["span_id"], "pred_start": c["pred_start"], "pred_end": c["pred_end"], "pred_duration": c["pred_duration"], "inference_score": c["inference_score"]}
            for c in cands
        ]
        top_clean = select_top_k_candidates_for_system(clean_cands, "system_c_mter", k=1)
        self.assertEqual(top_clean[0]["span_id"], "cand_high_score")

    def test_system_with_fewer_than_k_candidates_reported_accurately(self):
        """Verify that missing candidates are accurately recorded without fabrication."""
        csv_file = self.temp_dir / "test_missing.csv"
        rows = [
            {"episode_id": "ep_01", "system_mode": "system_a_transcript_only", "span_id": "a1", "pred_start": 10.0, "pred_end": 40.0, "pred_duration": 30.0, "inference_score": 0.85},
            # system_b is missing entirely for ep_01
            {"episode_id": "ep_01", "system_mode": "system_c_mter", "span_id": "c1", "pred_start": 20.0, "pred_end": 50.0, "pred_duration": 30.0, "inference_score": 0.75},
            {"episode_id": "ep_01", "system_mode": "system_d_mter_refined", "span_id": "d1", "pred_start": 19.8, "pred_end": 50.2, "pred_duration": 30.4, "inference_score": 0.75},
        ]
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        out_pilot = self.temp_dir / "pilot_out"
        res = export_top1_pilot_package(str(csv_file), str(out_pilot))

        with open(out_pilot / "pilot_system_comparison.json", "r", encoding="utf-8") as f:
            comparison = json.load(f)

        self.assertFalse(comparison["all_combinations_present"])
        self.assertEqual(comparison["missing_candidates_count"], 1)
        self.assertEqual(comparison["missing_candidates"][0]["system_mode"], "system_b_simple_aggregation")
        self.assertEqual(comparison["system_summaries"]["system_b_simple_aggregation"]["total_clips"], 0)
        self.assertEqual(comparison["system_summaries"]["system_a_transcript_only"]["total_clips"], 1)

    def test_system_c_and_d_base_candidate_identity(self):
        """Verify that System C and System D top-1 candidates evaluate the identical base candidate."""
        csv_path = Path("runs/rhapsody_real_pilot_5ep_run/timestamp_comparison.csv")
        if not csv_path.exists():
            csv_path = Path(__file__).parent.parent / "runs" / "rhapsody_real_pilot_5ep_run" / "timestamp_comparison.csv"

        if csv_path.exists():
            out_pilot = self.temp_dir / "test_cd_pilot"
            export_top1_pilot_package(str(csv_path), str(out_pilot))

            with open(out_pilot / "pilot_system_comparison.json", "r", encoding="utf-8") as f:
                comparison = json.load(f)

            for ep_id, sys_map in comparison["per_episode_breakdown"].items():
                c_data = sys_map["system_c_mter"]
                d_data = sys_map["system_d_mter_refined"]

                # Ensure overlap is near 100% (within refinement boundary deltas < 1.0s)
                c_s, c_e = c_data["start_sec"], c_data["end_sec"]
                d_s, d_e = d_data["start_sec"], d_data["end_sec"]
                overlap = max(0.0, min(c_e, d_e) - max(c_s, d_s))
                union = max(c_e, d_e) - min(c_s, d_s)
                iou = overlap / union if union > 0 else 0.0

                self.assertGreater(
                    iou,
                    0.95,
                    f"System C and D do not share identical base interval on {ep_id}: C=[{c_s}, {c_e}], D=[{d_s}, {d_e}]",
                )

    def test_ui_rank_filename_and_manifest_consistency(self):
        """Verify that UI rank, manifest rank, and rendered filename format agree consistently."""
        # Test candidate sorting
        c1 = MTERCandidate(
            candidate_id="cand_1",
            proposed_start=1.00,
            proposed_end=3.00,
            duration=2.00,
            confidence_estimate=0.496,
            contributing_experts=["transcript", "visual"],
            temporal_agreement_iou=0.487,
            decision_summary="Decision summary cand 1",
            semantic_verification_status="passed",
            semantic_verification=SemanticVerificationCriteria(
                natural_semantic_start=True,
                sufficient_context=True,
                important_content_retained=True,
                complete_conversational_payoff=True,
                no_mid_sentence_ending=True,
            ),
        )
        c2 = MTERCandidate(
            candidate_id="cand_2",
            proposed_start=2.00,
            proposed_end=4.50,
            duration=2.50,
            confidence_estimate=0.477,
            contributing_experts=["transcript", "prosody"],
            temporal_agreement_iou=0.526,
            decision_summary="Decision summary cand 2",
            semantic_verification_status="passed",
            semantic_verification=SemanticVerificationCriteria(
                natural_semantic_start=True,
                sufficient_context=True,
                important_content_retained=True,
                complete_conversational_payoff=True,
                no_mid_sentence_ending=True,
            ),
        )

        candidates = [c2, c1]  # Pass in reverse order
        sorted_cands = sorted(candidates, key=canonical_candidate_sort_key, reverse=True)

        # Higher confidence (0.496 vs 0.477) must rank #1
        self.assertEqual(sorted_cands[0].candidate_id, "cand_1")
        self.assertEqual(sorted_cands[1].candidate_id, "cand_2")

        # Filename format verification
        renderer = VideoRenderer()
        manifest = renderer.render_highlights(
            source_video=Path(__file__).parent / "sample_short.mp4",
            candidates=candidates,
            output_dir=self.temp_dir,
            top_k=2,
        )

        self.assertEqual(len(manifest.clips), 2)
        clip1 = manifest.clips[0]
        clip2 = manifest.clips[1]

        # Rank 1 must be cand_1
        self.assertEqual(clip1.rank, 1)
        self.assertEqual(clip1.candidate_id, "cand_1")
        self.assertIn("highlight_01_1.00_3.00.mp4", Path(clip1.output_path).name)
        self.assertEqual(clip1.score, 0.496)

        # Rank 2 must be cand_2
        self.assertEqual(clip2.rank, 2)
        self.assertEqual(clip2.candidate_id, "cand_2")
        self.assertIn("highlight_02_2.00_4.50.mp4", Path(clip2.output_path).name)
        self.assertEqual(clip2.score, 0.477)

    def test_manifest_stores_refined_and_fallback_fields(self):
        """Verify that ClipsManifest accurately stores refined timestamps and fallback reasons."""
        c = MTERCandidate(
            candidate_id="cand_refined",
            proposed_start=1.00,
            proposed_end=3.00,
            duration=2.00,
            confidence_estimate=0.518,
            contributing_experts=["transcript", "conversation"],
            temporal_agreement_iou=0.538,
            decision_summary="Decision summary cand refined",
            semantic_verification_status="passed",
            semantic_verification=SemanticVerificationCriteria(
                natural_semantic_start=True,
                sufficient_context=True,
                important_content_retained=True,
                complete_conversational_payoff=True,
                no_mid_sentence_ending=True,
            ),
            boundary_refinement=BoundaryRefinementMetadata(
                refined=True,
                original_start=1.20,
                original_end=2.90,
                original_duration=1.70,
                refined_start=1.00,
                refined_end=3.00,
                refined_duration=2.00,
                start_delta_sec=-0.20,
                end_delta_sec=0.10,
                start_evidence_type="sentence_boundary",
                end_evidence_type="sentence_boundary",
                signals_used=["transcript_sentence_boundary"],
                rationale="Snapped to sentence boundaries",
                fallback_reason=None,
            ),
        )
        renderer = VideoRenderer()
        manifest = renderer.render_highlights(
            source_video=Path(__file__).parent / "sample_short.mp4",
            candidates=[c],
            output_dir=self.temp_dir,
            top_k=1,
        )

        clip = manifest.clips[0]
        self.assertEqual(clip.refinement_status, "refined")
        self.assertIsNone(clip.fallback_reason)
        self.assertEqual(clip.start_time, 1.00)
        self.assertEqual(clip.end_time, 3.00)
        self.assertEqual(clip.original_start, 1.20)
        self.assertEqual(clip.original_end, 2.90)
        self.assertEqual(clip.score, 0.518)

    def test_blank_human_rating_fields(self):
        """Verify that all score fields in the exported rating sheet remain completely blank."""
        pilot_sheet = Path("evaluation/human_eval_top1_pilot/human_eval_rating_sheet.csv")
        if not pilot_sheet.exists():
            pilot_sheet = Path(__file__).parent.parent / "evaluation" / "human_eval_top1_pilot" / "human_eval_rating_sheet.csv"

        self.assertTrue(pilot_sheet.exists(), "human_eval_rating_sheet.csv not found")
        with open(pilot_sheet, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 60)  # 20 clips * 3 raters
        score_fields = [
            "score_interestingness",
            "score_self_containment",
            "score_content_payoff",
            "score_start_end_quality",
            "score_redundancy",
            "score_shareability",
            "rater_comments",
        ]
        for idx, row in enumerate(rows):
            for field in score_fields:
                val = row.get(field, "")
                self.assertEqual(
                    val,
                    "",
                    f"Row {idx} has non-blank rating field '{field}': '{val}'",
                )


if __name__ == "__main__":
    unittest.main()
