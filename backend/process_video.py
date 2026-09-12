"""
ClipSense End-to-End Processing CLI (W1 -> W5)

Processes a video through the complete multimodal pipeline:
1. W2: Multimodal Extraction (WhisperX transcription, frames, acoustic prosody, speaker turns)
2. W3: Semantic Experts (Transcript Expert & Conversation Expert)
3. W4: Physical Experts (Visual Expert & Prosody Expert)
4. W5: Multimodal Temporal Evidence Reasoner (MTER)

Usage:
    python process_video.py <path_to_video.mp4> [--run-id <id>] [--device cpu|cuda] [--force-rerun]
"""

import argparse
import json
import logging
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure backend root is on sys.path
backend_dir = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(backend_dir))

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.config import config
from core.llm import LLMClient, get_llm_client
from core.schemas import (
    CommonEvidenceBundle,
    ConversationData,
    ExpertEvidenceBundle,
    MTEROutput,
    ProsodyData,
    TranscriptData,
    VisualData,
)
from pipeline.experts.conversation_expert import (
    ConversationExpert,
    ConversationLLMCandidate,
    ConversationLLMResponse,
)
from pipeline.experts.prosody_expert import ProsodyExpert
from pipeline.experts.transcript_expert import (
    TranscriptExpert,
    TranscriptLLMResponse,
    TranscriptSemanticCandidate,
)
from pipeline.experts.visual_expert import VisualExpert
from pipeline.extraction.extractor_pipeline import MultimodalExtractionPipeline
from pipeline.extraction.transcript_extractor import TranscriptExtractor
from pipeline.mter.evidence_bundle import build_common_evidence_bundle
from pipeline.mter.reasoner import MTERReasoner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("clipsense.cli")


class PipelineMockLLMClient(LLMClient):
    """
    Offline/deterministic mock LLM client for fast end-to-end pipeline execution
    without external API keys or network rate limits.
    """

    def generate_structured(
        self,
        prompt: str,
        response_schema: Any,
        system_instruction: Optional[str] = None,
    ) -> Any:
        import re

        if response_schema == TranscriptLLMResponse:
            indices = [int(m) for m in re.findall(r"\[(\d+)\]", prompt)]
            candidates = []
            if len(indices) >= 15:
                start_i = indices[min(3, len(indices) - 1)]
                end_i = indices[min(len(indices) - 3, 35)]
                candidates.append(
                    TranscriptSemanticCandidate(
                        start_word_index=start_i,
                        end_word_index=end_i,
                        evidence_type="explanatory_claim",
                        explanation="Localized thematic claim extracted from spoken dialogue window.",
                        confidence_estimate=0.88,
                        topic_transition=False,
                        contextual_completeness=0.86,
                        semantic_importance=0.84,
                        self_contained=True,
                    )
                )
            return TranscriptLLMResponse(candidates=candidates)

        elif response_schema == ConversationLLMResponse:
            time_matches = re.findall(r"\[([\d\.]+)s -> ([\d\.]+)s\]", prompt)
            candidates = []
            if time_matches:
                cur_s = float(time_matches[0][0])
                for sm, em in time_matches:
                    st = float(sm)
                    et = float(em)
                    if et - cur_s >= 20.0:
                        candidates.append(
                            ConversationLLMCandidate(
                                start_time_estimate=cur_s,
                                end_time_estimate=min(et, cur_s + 40.0),
                                evidence_type="monologue_thematic_unit",
                                explanation="Thematic discourse progression unit across adjacent speech segments.",
                                confidence_estimate=0.87,
                                discourse_unit_complete=True,
                                speaker_interaction=False,
                                pause_boundary_supported=True,
                                discourse_phase="setup_development_payoff",
                            )
                        )
                        cur_s = et
                        if len(candidates) >= 5:
                            break
                if not candidates and len(time_matches) > 0:
                    candidates.append(
                        ConversationLLMCandidate(
                            start_time_estimate=float(time_matches[0][0]),
                            end_time_estimate=min(float(time_matches[-1][1]), float(time_matches[0][0]) + 30.0),
                            evidence_type="monologue_thematic_unit",
                            explanation="Initial cohesive monologue segment with clear setup and payoff.",
                            confidence_estimate=0.85,
                            discourse_unit_complete=True,
                            speaker_interaction=False,
                            pause_boundary_supported=True,
                            discourse_phase="setup_development_payoff",
                        )
                    )
            return ConversationLLMResponse(candidates=candidates)

        return response_schema()


def process_video(
    video_path: str,
    run_id: str = None,
    device: str = "cpu",
    force_rerun: bool = False,
    whisper_model: str = "tiny.en",
    mock_llm: bool = False,
    llm_model: Optional[str] = None,
    window_duration: Optional[float] = None,
):
    video_p = pathlib.Path(video_path).resolve()
    if not video_p.exists():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")

    if not run_id:
        run_id = video_p.stem

    runs_dir = backend_dir / "runs"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print("CLIPSENSE MULTIMODAL VIDEO PROCESSING PIPELINE (W1 -> W5)")
    print(f"Input Video: {video_p.name} ({video_p.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Run ID:      {run_id}")
    print(f"Artifacts:   {run_dir}")
    print("=" * 75)

    start_total = time.time()

    # -------------------------------------------------------------------------
    # STAGE 1 (W2): MULTIMODAL FEATURE EXTRACTION
    # -------------------------------------------------------------------------
    print("\n[Stage 1/4] Running Multimodal Feature Extraction (W2)...")
    t_file = run_dir / "transcript.json"
    c_file = run_dir / "conversation.json"
    v_file = run_dir / "frames.json"
    p_file = run_dir / "prosody.json"
    m_file = run_dir / "manifest.json"

    needs_extraction = force_rerun or not (t_file.exists() and c_file.exists() and v_file.exists() and p_file.exists())

    if needs_extraction:
        transcript_extractor = TranscriptExtractor(
            model_name=whisper_model,
            device=device,
            compute_type="int8" if device == "cpu" else "float16",
            strict_mode=True,
        )
        extraction_pipeline = MultimodalExtractionPipeline(
            transcript_extractor=transcript_extractor,
        )
        artifacts = extraction_pipeline.run(
            video_path=str(video_p),
            run_id=run_id,
            runs_base_dir=str(runs_dir),
            force_rerun=force_rerun,
        )
    else:
        print("  -> Existing W2 extraction artifacts found in run directory. Reusing cached extraction.")

    with open(t_file, "r", encoding="utf-8") as f:
        transcript_data = TranscriptData.model_validate_json(f.read())
    with open(c_file, "r", encoding="utf-8") as f:
        conversation_data = ConversationData.model_validate_json(f.read())
    with open(v_file, "r", encoding="utf-8") as f:
        visual_data = VisualData.model_validate_json(f.read())
    with open(p_file, "r", encoding="utf-8") as f:
        prosody_data = ProsodyData.model_validate_json(f.read())

    total_duration = visual_data.duration
    print(f"  [✓] Transcript loaded ({len(transcript_data.words)} words, duration: {transcript_data.duration:.2f}s)")
    print(f"  [✓] Conversation loaded ({len(conversation_data.turns)} turns/segments)")
    print(f"  [✓] Visual loaded ({len(visual_data.sampled_frames)} frames, {len(visual_data.scene_boundaries)} cuts)")
    print(f"  [✓] Prosody loaded ({len(prosody_data.windows)} acoustic windows)")

    # -------------------------------------------------------------------------
    # STAGE 2 (W3): SEMANTIC EXPERTS
    # -------------------------------------------------------------------------
    print("\n[Stage 2/4] Generating Semantic Evidence Proposals (W3)...")
    t_ev_file = run_dir / "transcript_evidence.json"
    c_ev_file = run_dir / "conversation_evidence.json"

    if force_rerun or not (t_ev_file.exists() and c_ev_file.exists()):
        has_api_key = bool(os.getenv("GEMINI_API_KEY") or config.gemini_api_key)
        if mock_llm:
            print("  -> Using mock LLM mode (fast deterministic candidates, no external API calls).")
            llm_client = PipelineMockLLMClient()
        elif not has_api_key:
            print("  -> GEMINI_API_KEY not found in environment. Using fallback PipelineMockLLMClient.")
            llm_client = PipelineMockLLMClient()
        else:
            llm_cfg = config.llm.model_copy()
            if llm_model:
                llm_cfg.model_name = llm_model
            print(f"  -> Using live Gemini LLM ({llm_cfg.model_name}) with automatic rate-limit backoff.")
            llm_client = get_llm_client(llm_config=llm_cfg)

        t_expert = TranscriptExpert(
            llm_client=llm_client,
            config=config.transcript,
            window_duration_sec=window_duration,
        )
        t_evidence = t_expert.evaluate(transcript_data)
        with open(t_ev_file, "w", encoding="utf-8") as f:
            f.write(t_evidence.model_dump_json(indent=2))

        c_expert = ConversationExpert(
            llm_client=llm_client,
            config=config.conversation,
        )
        c_evidence = c_expert.evaluate(conversation_data, context={"transcript_data": transcript_data})
        with open(c_ev_file, "w", encoding="utf-8") as f:
            f.write(c_evidence.model_dump_json(indent=2))
    else:
        print("  -> Existing W3 semantic evidence artifacts found. Reusing cached proposals.")
        with open(t_ev_file, "r", encoding="utf-8") as f:
            t_evidence = ExpertEvidenceBundle.model_validate_json(f.read())
        with open(c_ev_file, "r", encoding="utf-8") as f:
            c_evidence = ExpertEvidenceBundle.model_validate_json(f.read())

    print(f"  [✓] Transcript Expert:   {len(t_evidence.proposals)} temporal proposals generated")
    print(f"  [✓] Conversation Expert: {len(c_evidence.proposals)} temporal proposals generated")

    # -------------------------------------------------------------------------
    # STAGE 3 (W4): PHYSICAL EXPERTS
    # -------------------------------------------------------------------------
    print("\n[Stage 3/4] Generating Physical Sensory Evidence Proposals (W4)...")
    v_ev_file = run_dir / "visual_evidence.json"
    p_ev_file = run_dir / "prosody_evidence.json"

    if force_rerun or not (v_ev_file.exists() and p_ev_file.exists()):
        v_expert = VisualExpert(config=config.visual)
        v_evidence = v_expert.evaluate(visual_data)
        with open(v_ev_file, "w", encoding="utf-8") as f:
            f.write(v_evidence.model_dump_json(indent=2))

        p_expert = ProsodyExpert(config=config.prosody)
        p_evidence = p_expert.evaluate(prosody_data)
        with open(p_ev_file, "w", encoding="utf-8") as f:
            f.write(p_evidence.model_dump_json(indent=2))
    else:
        print("  -> Existing W4 physical evidence artifacts found. Reusing cached proposals.")
        with open(v_ev_file, "r", encoding="utf-8") as f:
            v_evidence = ExpertEvidenceBundle.model_validate_json(f.read())
        with open(p_ev_file, "r", encoding="utf-8") as f:
            p_evidence = ExpertEvidenceBundle.model_validate_json(f.read())

    print(f"  [✓] Visual Expert:  {len(v_evidence.proposals)} temporal proposals generated")
    print(f"  [✓] Prosody Expert: {len(p_evidence.proposals)} temporal proposals generated")

    # -------------------------------------------------------------------------
    # STAGE 4 (W5): MULTIMODAL TEMPORAL EVIDENCE REASONER (MTER)
    # -------------------------------------------------------------------------
    print("\n[Stage 4/4] Executing Multimodal Temporal Evidence Reasoner (W5)...")
    bundle = build_common_evidence_bundle(
        video_id=run_id,
        total_duration=total_duration,
        transcript_evidence=t_evidence,
        visual_evidence=v_evidence,
        prosody_evidence=p_evidence,
        conversation_evidence=c_evidence,
    )

    reasoner = MTERReasoner(config.mter)
    output, ledger = reasoner.reason(bundle)

    # Save artifacts
    out_file = run_dir / "mter_output.json"
    ledger_file = run_dir / "mter_ledger.json"

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(output.model_dump_json(indent=2))
    with open(ledger_file, "w", encoding="utf-8") as f:
        f.write(ledger.model_dump_json(indent=2))

    total_time = time.time() - start_total
    print(f"  [✓] CommonEvidenceBundle aggregated (total proposals: {ledger.total_proposals})")
    print(f"  [✓] Temporal compatibility graph: {len(ledger.event_regions)} event region(s) formed")
    print(f"  [✓] Boundary clusters: {len(ledger.start_clusters)} start, {len(ledger.end_clusters)} end clusters")
    print(f"  [✓] Cross-modal conflicts: {len(ledger.conflicts)} detected across regions")
    print(f"  [✓] Evaluated candidates: {len(ledger.evaluated_candidates)} intervals evaluated")

    # -------------------------------------------------------------------------
    # FINAL DISPLAY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("CLIPSENSE W5: FINAL HIGHLIGHT SELECTION & AUDIT RESULT")
    print("=" * 75)

    if not output.candidates:
        print("No valid candidate satisfied temporal duration constraints.")
        print(f"Ledger Summary: {ledger.decision_summary}")
        return

    top = output.candidates[0]
    m = top.support_metrics

    print(f"\n  SELECTED HIGHLIGHT SPAN:")
    print(f"    Timestamps:          [{top.proposed_start:.3f}s -> {top.proposed_end:.3f}s]")
    print(f"    Duration:            {top.duration:.2f} seconds")
    print(f"    Evidence Strength:   {top.confidence_estimate:.3f} (Composite score on [0.0, 1.0])")
    print(f"    Temporal Agreement:  {top.temporal_agreement_iou:.3f} mean active IoU")

    print(f"\n  MODALITY SUPPORT BREAKDOWN:")
    print(f"    Evidence Present:    {len(top.modalities_present)}/4 ({', '.join(top.modalities_present)})")
    print(f"    Strong Support:      {', '.join(top.strong_support_modalities) if top.strong_support_modalities else 'None'}")
    print(f"    Moderate Support:    {', '.join(top.moderate_support_modalities) if top.moderate_support_modalities else 'None'}")
    print(f"    Weak/Secondary:      {', '.join(top.weak_support_modalities) if top.weak_support_modalities else 'None'}")
    print(f"    Coverage per Modality:")
    print(f"      - Transcript:      coverage={m.transcript_coverage:.2f}, support={m.transcript_support:.2f}, IoU={m.transcript_iou:.2f}")
    print(f"      - Conversation:    coverage={m.conversation_coverage:.2f}, support={m.conversation_support:.2f}, IoU={m.conversation_iou:.2f}")
    print(f"      - Visual:          coverage={m.visual_coverage:.2f}, support={m.visual_support:.2f}, IoU={m.visual_iou:.2f}")
    print(f"      - Prosody:         coverage={m.prosody_coverage:.2f}, support={m.prosody_support:.2f}, IoU={m.prosody_iou:.2f}")

    print(f"\n  BOUNDARY EVIDENCE:")
    print(f"    Linguistic Support:  {top.linguistic_boundary_support:.2f} (spoken word/segment alignment)")
    print(f"    Activity Support:    {top.boundary_activity_support:.2f} (scene transitions/vocal emphasis)")

    print(f"\n  CONFLICT AUDIT (SCOPED TO CANDIDATE):")
    print(f"    Detected in Region:  {top.detected_conflicts_count}")
    print(f"    Affecting Candidate: {top.conflicts_affecting_candidate_count}")
    print(f"    Unresolved:          {top.unresolved_conflicts_count}")
    if ledger.conflicts_affecting_candidate:
        for c in ledger.conflicts_affecting_candidate:
            print(f"      * [{c.boundary_type.upper()}] {c.description}")
    else:
        print("      * No unresolved boundary conflicts directly affect this candidate.")

    print(f"\n  DECISION SUMMARY:")
    print(f"    {top.decision_summary}")

    print(f"\n  GENERATED ARTIFACTS SAVED TO DISK:")
    print(f"    - Reasoning Output: {out_file}")
    print(f"    - Complete Ledger:  {ledger_file}")
    print(f"\nTotal Pipeline Execution Time: {total_time:.2f}s")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="ClipSense End-to-End Multimodal Video Processor (W1 -> W5)")
    parser.add_argument("video_path", type=str, help="Path to input video file (.mp4, .mov, etc.)")
    parser.add_argument("--run-id", type=str, default=None, help="Identifier for the run output directory")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Computation device for WhisperX")
    parser.add_argument("--whisper-model", type=str, default="tiny.en", help="Whisper model size (tiny.en, base.en, large-v2)")
    parser.add_argument("--force-rerun", action="store_true", help="Force re-extraction even if artifacts exist")
    parser.add_argument("--mock-llm", action="store_true", help="Use deterministic mock LLM for fast offline processing without API rate limits")
    parser.add_argument("--llm-model", type=str, default=None, help="LLM model variant (e.g., gemini-2.0-flash, gemini-1.5-flash, gemini-2.5-flash)")
    parser.add_argument("--window-duration", type=float, default=None, help="Custom bounded analysis window in seconds for transcript expert")

    args = parser.parse_args()
    process_video(
        video_path=args.video_path,
        run_id=args.run_id,
        device=args.device,
        force_rerun=args.force_rerun,
        whisper_model=args.whisper_model,
        mock_llm=args.mock_llm,
        llm_model=args.llm_model,
        window_duration=args.window_duration,
    )


if __name__ == "__main__":
    main()
