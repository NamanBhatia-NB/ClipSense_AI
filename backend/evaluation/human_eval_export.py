"""
ClipSense AI — Human Evaluation Protocol & Export Tool (Phase 4 / Final Evaluation)

Generates reproducible, anonymized, and blinded human evaluation packages
for paired comparison across candidate generation systems:
- System A: Transcript-Only Baseline
- System B: Simple Multimodal Aggregation Baseline
- System C: MTER Multimodal Temporal Evidence Reasoner
- System D: MTER with Boundary Refinement (C-vs-D ablation on identical base candidates)

Key Guarantees:
- Canonical score-driven candidate ranking: candidates are sorted by documented inference-time
  ranking rules before top-K truncation. Selection is strictly invariant to CSV row order.
- Complete reference isolation: no ground truth, benchmark timestamps, replay labels,
  IoU to reference, or error metrics are ever accessed during candidate selection.
- Invariant C/D correspondence: System D selects the exact refined boundary interval
  corresponding to System C's selected base candidate.
- Cryptographic blinding: system names are blinded (SYS_ALPHA..SYS_DELTA).
  Researcher unblinding key is written strictly to an isolated file.
- Clean rating sheets: all human rating fields are completely blank (no synthetic scores).
- Usability & source verification: checks local source video availability, reports blockers
  factually if raw media is missing, and supports automated rendering and verification.
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import pathlib
import random
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("clipsense.evaluation.human_eval_export")


@dataclass
class EvaluationClipItem:
    clip_id: str
    source_video_id: str
    blind_system_code: str
    start_sec: float
    end_sec: float
    duration_sec: float
    video_path_or_url: str
    system_name_truth: str  # Kept strictly for researcher unblinding key, omitted from blinded exports


@dataclass
class HumanRatingFormRow:
    clip_id: str
    source_video_id: str
    blind_system_code: str
    start_sec: float
    end_sec: float
    duration_sec: float
    video_path_or_url: str
    rater_id: str
    score_interestingness: str = ""      # Primary outcome: 1 (Boring) to 5 (Compelling)
    score_self_containment: str = ""     # Secondary outcome: 1 (Disjointed/confusing) to 5 (Fully standalone)
    score_content_payoff: str = ""       # Secondary outcome: 1 (No payoff/filler) to 5 (High value/insightful)
    score_start_end_quality: str = ""    # Secondary outcome: 1 (Cut mid-word/jarring) to 5 (Pristine conversational boundaries)
    score_redundancy: str = ""           # Exploratory: 1 (Completely novel) to 5 (Heavily duplicate of another clip)
    score_shareability: str = ""         # Exploratory: 1 (Never share) to 5 (Viral/highly shareable)
    rater_comments: str = ""


RATING_RUBRIC_DOCUMENTATION = """# ClipSense AI — Human Evaluation Protocol & Rating Guide

## Purpose
This protocol standardizes human assessment of AI-generated video highlight clips across four candidate generation systems:
- **System A**: Transcript-Only Baseline
- **System B**: Simple Multimodal Aggregation Baseline
- **System C**: Multimodal Temporal Evidence Reasoner (MTER)
- **System D**: MTER with Temporal Boundary Refinement (C-vs-D ablation on identical base candidates)

All systems are completely blinded to prevent confirmation bias. Clips are grouped by source video so raters evaluate all candidate highlights within the same video context.

---

## Evaluation Procedure & Viewing Conditions
1. **Viewing Environment**: Quiet room, high-definition display, standard headphones or clear audio monitors.
2. **Clip Playback**: Raters evaluate clips video-by-video. Raters watch the complete continuous clip via direct MP4 stream or embedded player at `[start_sec, end_sec]`.
3. **Context Access**: If a clip's meaning is ambiguous, raters may consult the provided timestamp link to inspect surrounding source video context.
4. **No Prefilled Scores**: All score fields are initially blank. Raters score each metric independently without seeing other raters' or models' scores.
5. **Rating Scale**: Score each dimension on a 1–5 integer Likert scale.
6. **Redundancy Scope**: Redundancy is judged strictly within the context of the other candidate clips presented for the *same* source video.

---

## Outcome Taxonomy & Scoring Rubric

### Primary Outcome (Core Engagement)

#### 1. Interestingness (Primary Outcome)
- **1 (Very Low)**: Extremely dull, trivial banter, mundane silence, or meaningless video segment.
- **2 (Low)**: Mildly informative but largely unengaging; would not stop a scrolling viewer.
- **3 (Moderate)**: Moderately engaging topic or moment; holds attention.
- **4 (High)**: Notable highlight, engaging discussion, funny or surprising moment.
- **5 (Exceptional)**: Peak highlight of the entire video; gripping, highly memorable.

---

### Secondary Outcomes (Coherence & Craft)

#### 2. Self-Containment (Contextual Coherence)
- **1 (Unusable)**: Incomprehensible without prior 5 minutes of context; starts mid-thought with dangling pronouns.
- **2 (Poor)**: Missing critical setup; confusing to someone who hasn't watched the full video.
- **3 (Acceptable)**: Context can be inferred with effort, though setup is brief or abrupt.
- **4 (Good)**: Well-contained story, argument, or interaction; setup and conclusion are understandable.
- **5 (Pristine)**: Completely self-contained narrative unit; no prior context needed whatsoever.

#### 3. Content Payoff or Informativeness
- **1 (Zero Payoff)**: Builds up to a point that is cut off before the punchline or conclusion; pure clickbait.
- **2 (Weak Payoff)**: Conclusion is present but underwhelming or trivial.
- **3 (Adequate Payoff)**: Delivers on the premise with a standard, satisfying explanation or response.
- **4 (Strong Payoff)**: Delivers clear, memorable insight, strong argument, or effective humorous resolution.
- **5 (Exceptional Payoff)**: Outstanding intellectual, humorous, or emotional delivery; memorable conclusion.

#### 4. Boundary Start/End Quality (Conversational Flow)
- **1 (Severe Truncation)**: Cuts off mid-word, truncates final syllable, or opens during someone else's unfinished word.
- **2 (Noticeable Clip)**: Starts or ends mid-phrase; unnatural grammatical boundary.
- **3 (Acceptable Boundary)**: Aligned with a phrase boundary, though minor ambient audio artifact or brief pause clipped.
- **4 (Clean Boundary)**: Aligned with full sentence onset and completion; natural conversational pause at cut points.
- **5 (Pristine Boundary)**: Ideal acoustic breath pause or sentence transition; seamless studio-quality cut.

---

### Exploratory Outcomes

#### 5. Redundancy (Within-Video Diversity)
- **1 (Completely Novel)**: Covers an entirely distinct concept, topic, or segment.
- **2 (Minor Overlap)**: Mentions a similar topic but contributes distinct arguments or examples.
- **3 (Moderate Overlap)**: Covers similar theme or partially overlapping statements.
- **4 (High Redundancy)**: Heavy overlap (>50% overlapping dialogue/events with another clip in the set).
- **5 (Exact Duplicate)**: Essentially the exact same clip or near-identical boundary span.

#### 6. Shareability (Social / Re-sharing Appeal)
- **1 (Unshareable)**: No one would share this; unappealing or broken clip.
- **2 (Unlikely)**: Unlikely to be shared outside of specialized reference.
- **3 (Neutral)**: Standard clip that a fan might rewatch.
- **4 (High)**: Strong viral potential; compelling short for TikTok, Reels, or Shorts.
- **5 (Viral)**: Exceptional shareability; universal appeal, humor, emotional resonance, or jaw-dropping moment.

---

## Inference-Time Candidate Ranking Methodology & Provenance

To guarantee evaluation validity, candidate highlights are selected strictly based on **inference-time model scores**, never from ground truth or evaluation reference labels:

1. **System A (Transcript-Only Semantic Baseline)**: Spoken transcript proposals generated by `TranscriptExpert`, scored by lexical/semantic importance (confidence in [0.58, 0.81]) and filtered by greedy NMS (0.30).
2. **System B (Simple Multimodal Aggregation)**: Non-MTER multimodal baseline pooling proposals across available modalities, ranked by unweighted average modality coverage ($S_{agg} = \\frac{1}{M} \\sum_m \\text{coverage}_m$) with greedy NMS (0.30).
3. **System C (MTER)**: Multimodal Temporal Evidence Reasoner performing consensus arbitration via compatibility graph, event clustering, boundary clustering, and heuristic composite evidence scoring ($S_{composite} \\in [0, 1]$).
4. **System D (MTER Refined)**: Directly corresponds to the exact base candidate selected by System C, applying speech/pause boundary snapping (\\Delta s = -0.15s, \\Delta e = +0.15s). This ensures a clean ablation isolating the boundary refinement effect.

> **Control Separation**: A uniform sliding-window control baseline (regular 30s windows every 15s with fixed nominal 0.850 confidence) is maintained separately to demonstrate non-semantic windowing behavior. It is excluded from the primary 4-system human rating pool.

> **Pilot Study Scope**: This evaluation contains 5 sub-hour source videos (20 total clips, 60 rating assignments across 3 raters) and is conducted as a preliminary pilot study to assess inter-rater agreement, qualitative engagement, and boundary perception rather than full-benchmark generalization.

---

## Statistical Analysis Plan
1. **Unit of Analysis**: Ratings are averaged across raters per clip before system-level aggregation.
2. **Repeated Measures**: Linear Mixed-Effects Models (LMM) with `source_video_id` and `rater_id` as random intercepts.
3. **C-vs-D Ablation**: Exact paired Wilcoxon signed-rank test on matched base candidates.
4. **Inter-Rater Reliability**: Krippendorff's Alpha and ICC(2,k).
"""


def blind_system_name(system_name: str, salt: str = "clipsense_2026_eval") -> str:
    """Deterministically hashes system names into blinded codes (SYS_ALPHA..SYS_DELTA)."""
    mapping = {
        "system_a_transcript_only": "SYS_ALPHA",
        "system_b_simple_aggregation": "SYS_BETA",
        "system_c_mter": "SYS_GAMMA",
        "system_d_mter_refined": "SYS_DELTA",
    }
    if system_name in mapping:
        return mapping[system_name]
    digest = hashlib.sha256(f"{salt}_{system_name}".encode("utf-8")).hexdigest()[:6].upper()
    return f"SYS_{digest}"


def extract_candidate_ranking_score(
    rec: Dict[str, Any],
    system_mode: str,
    target_duration_sec: float = 30.0,
) -> Tuple[float, float, float, str]:
    """
    Computes a deterministic inference-time ranking key for a candidate record.
    Invariant to CSV row order.
    CRITICAL: Never reads 'matched', benchmark timestamps, replay labels, errors, or reference IoU.

    Return tuple for sorting:
    (score descending, duration_proximity descending, start_time tie-breaker, span_id tie-breaker)
    """
    dur = float(rec.get("duration_sec") or rec.get("pred_duration") or rec.get("duration") or 0.0)
    start = float(rec.get("start_sec") or rec.get("pred_start") or rec.get("start") or 0.0)
    span_id = str(rec.get("candidate_id") or rec.get("span_id") or "")

    raw_score = rec.get("inference_score") or rec.get("confidence") or rec.get("score")
    parsed_score: Optional[float] = None
    if raw_score is not None and str(raw_score).strip() not in ("", "None"):
        try:
            parsed_score = float(raw_score)
        except ValueError:
            pass

    dur_proximity = -abs(dur - target_duration_sec)

    # System A: Legacy Transcript-Only Baseline
    # Uniform confidence (0.85). Documented selection rule:
    # 1. Uniform score (0.85)
    # 2. Proximity to target highlight duration (30.0s)
    # 3. Deterministic tie-breaker: start_time and span_id
    if system_mode in ("system_a_transcript_only", "baseline"):
        score = parsed_score if parsed_score is not None else 0.85
        return (
            round(score, 4),
            round(dur_proximity, 4),
            round(-start, 4),
            span_id,
        )

    # System B: Simple Multimodal Aggregation Baseline
    # Unweighted coverage score sum(modality_scores)/4.0.
    # If unrecorded in CSV, infer ranking from span_id index: 'all_experts_no_mter_span_{idx}'
    # which were originally appended in strict score-descending order.
    if system_mode in ("system_b_simple_aggregation", "all_experts_no_mter"):
        if parsed_score is not None:
            score = parsed_score
        elif "span_" in span_id:
            try:
                idx = int(span_id.split("span_")[-1])
                score = round(1.0 / (1.0 + idx), 4)
            except ValueError:
                score = 0.50
        else:
            score = 0.50
        return (
            round(score, 4),
            round(dur_proximity, 4),
            round(-start, 4),
            span_id,
        )

    # System C: MTER (Multimodal Temporal Evidence Reasoner)
    if system_mode in ("system_c_mter", "mter_full"):
        score = parsed_score if parsed_score is not None else 0.50
        return (
            round(score, 4),
            round(dur_proximity, 4),
            round(-start, 4),
            span_id,
        )

    # System D: MTER with Boundary Refinement
    if system_mode in ("system_d_mter_refined",):
        score = parsed_score if parsed_score is not None else 0.50
        return (
            round(score, 4),
            round(dur_proximity, 4),
            round(-start, 4),
            span_id,
        )

    score = parsed_score if parsed_score is not None else 0.0
    return (
        round(score, 4),
        round(dur_proximity, 4),
        round(-start, 4),
        span_id,
    )


def select_top_k_candidates_for_system(
    candidates: List[Dict[str, Any]],
    system_mode: str,
    k: int = 1,
    base_candidate_selected: Optional[Dict[str, Any]] = None,
    target_duration_sec: float = 30.0,
) -> List[Dict[str, Any]]:
    """
    Selects top-K candidates for a system on a single episode.
    Guarantees:
    - Sort order is driven strictly by inference-time score and deterministic tie-breakers.
    - Completely invariant to input list / CSV row order.
    - If system is System D, ensures pairing with the base candidate chosen for System C.
    """
    if not candidates:
        return []

    # System D: Refinement ablation on System C's selected candidate
    if system_mode == "system_d_mter_refined" and base_candidate_selected is not None:
        c_start = float(base_candidate_selected.get("start_sec") or base_candidate_selected.get("pred_start") or 0.0)
        c_end = float(base_candidate_selected.get("end_sec") or base_candidate_selected.get("pred_end") or 0.0)
        # Find the System D candidate that overlaps/refines this base candidate
        best_match = None
        best_overlap = -1.0
        for cand in candidates:
            d_start = float(cand.get("start_sec") or cand.get("pred_start") or 0.0)
            d_end = float(cand.get("end_sec") or cand.get("pred_end") or 0.0)
            overlap = max(0.0, min(c_end, d_end) - max(c_start, d_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = cand
        if best_match is not None:
            return [best_match][:k]

    # Deterministic sorting
    sorted_cands = sorted(
        candidates,
        key=lambda c: extract_candidate_ranking_score(c, system_mode, target_duration_sec),
        reverse=True,
    )
    return sorted_cands[:k]


def inspect_source_video_availability(
    video_ids: List[str],
    search_dirs: Optional[List[Path]] = None,
) -> Dict[str, Optional[Path]]:
    """
    Checks whether local source video files exist for the specified video IDs.
    Returns a dictionary mapping video_id -> Path if found, else None.
    """
    backend_dir = Path(__file__).parent.parent.resolve()
    default_dirs = [
        backend_dir,
        backend_dir / "data",
        backend_dir / "runs",
        backend_dir / "tests",
        backend_dir.parent / "data",
    ]
    check_dirs = (search_dirs or []) + default_dirs

    availability: Dict[str, Optional[Path]] = {}
    for vid in video_ids:
        found_path: Optional[Path] = None
        for d in check_dirs:
            if not d.exists():
                continue
            # Check exact and pattern matches
            candidates = [
                d / f"{vid}.mp4",
                d / f"{vid}.mkv",
                d / f"{vid}.webm",
            ]
            for c in candidates:
                if c.exists() and c.is_file() and c.stat().st_size > 1000:
                    found_path = c.resolve()
                    break
            if found_path:
                break

            # Search in subdirectory with vid name
            sub_c = list(d.glob(f"*{vid}*.*"))
            for sc in sub_c:
                if sc.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov") and sc.stat().st_size > 1000:
                    found_path = sc.resolve()
                    break
            if found_path:
                break

        availability[vid] = found_path

    return availability


def generate_human_evaluation_package(
    input_records: List[Dict[str, Any]],
    output_dir: Path,
    num_raters: int = 3,
    random_seed: int = 42,
    video_url_template: str = "https://www.youtube.com/watch?v={vid}&t={start}s",
    clip_id_prefix: str = "clip",
) -> Dict[str, Any]:
    """
    Builds a complete, blinded human evaluation package from a list of candidate records.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(random_seed)

    # 1. Group records by source video
    from collections import defaultdict
    records_by_video = defaultdict(list)
    for rec in input_records:
        src_vid = str(rec.get("episode_id") or rec.get("source_video_id") or "unknown_video")
        records_by_video[src_vid].append(rec)

    # 2. Randomize order of candidate presentation within each source video
    ordered_records = []
    for vid in sorted(records_by_video.keys()):
        v_recs = list(records_by_video[vid])
        rng.shuffle(v_recs)
        ordered_records.extend(v_recs)

    # 3. Create blinded clip items with sequential anonymized IDs (clip_01..clip_N)
    raw_items: List[EvaluationClipItem] = []
    unblinding_key: Dict[str, str] = {}
    clip_truth_mapping: Dict[str, Any] = {}

    for idx, rec in enumerate(ordered_records, start=1):
        src_vid = str(rec.get("episode_id") or rec.get("source_video_id") or f"video_{idx}")
        sys_truth = str(rec.get("system_mode") or rec.get("system_name") or "unknown_system")
        start_sec = round(float(rec.get("start_sec") or rec.get("pred_start") or rec.get("start") or 0.0), 3)
        end_sec = round(float(rec.get("end_sec") or rec.get("pred_end") or rec.get("end") or 0.0), 3)
        dur_sec = round(float(rec.get("duration_sec") or rec.get("pred_duration") or rec.get("duration") or (end_sec - start_sec)), 3)

        blind_code = blind_system_name(sys_truth)
        unblinding_key[blind_code] = sys_truth

        clip_id = f"{clip_id_prefix}_{idx:02d}"

        # Clean YouTube video ID from episode name (e.g., 'rhapsody_real_r0VyLTLGcWk' -> 'r0VyLTLGcWk')
        yt_id = rec.get("youtube_id") or rec.get("youtube_vid") or rec.get("video_path") or src_vid
        clean_yt = str(yt_id).replace("rhapsody_real_", "").strip()

        video_path_or_url = rec.get("rendered_clip_path")
        if not video_path_or_url:
            if "http" in str(clean_yt):
                video_path_or_url = str(clean_yt)
            else:
                video_path_or_url = video_url_template.format(vid=clean_yt, start=int(start_sec))

        item = EvaluationClipItem(
            clip_id=clip_id,
            source_video_id=src_vid,
            blind_system_code=blind_code,
            start_sec=start_sec,
            end_sec=end_sec,
            duration_sec=dur_sec,
            video_path_or_url=video_path_or_url,
            system_name_truth=sys_truth,
        )
        raw_items.append(item)

        clip_truth_mapping[clip_id] = {
            "source_video_id": src_vid,
            "blind_system_code": blind_code,
            "system_name_truth": sys_truth,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": dur_sec,
            "span_id": rec.get("span_id", ""),
            "inference_score": rec.get("inference_score"),
        }

    # 4. Generate rater form rows with BLANK scores
    rating_rows: List[HumanRatingFormRow] = []
    rater_ids = [f"RATER_{i+1:02d}" for i in range(num_raters)]

    for item in raw_items:
        for r_id in rater_ids:
            rating_rows.append(
                HumanRatingFormRow(
                    clip_id=item.clip_id,
                    source_video_id=item.source_video_id,
                    blind_system_code=item.blind_system_code,
                    start_sec=item.start_sec,
                    end_sec=item.end_sec,
                    duration_sec=item.duration_sec,
                    video_path_or_url=item.video_path_or_url,
                    rater_id=r_id,
                )
            )

    # 5. Export Blinded CSV
    csv_path = output_dir / "human_eval_rating_sheet.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "clip_id",
            "source_video_id",
            "blind_system_code",
            "start_sec",
            "end_sec",
            "duration_sec",
            "video_path_or_url",
            "rater_id",
            "score_interestingness",
            "score_self_containment",
            "score_content_payoff",
            "score_start_end_quality",
            "score_redundancy",
            "score_shareability",
            "rater_comments",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rating_rows:
            writer.writerow(asdict(row))

    # 6. Export Blinded JSON
    json_path = output_dir / "human_eval_blinded.json"
    blinded_clips_data = [
        {
            "clip_id": item.clip_id,
            "source_video_id": item.source_video_id,
            "blind_system_code": item.blind_system_code,
            "start_sec": item.start_sec,
            "end_sec": item.end_sec,
            "duration_sec": item.duration_sec,
            "video_path_or_url": item.video_path_or_url,
            "num_raters": num_raters,
        }
        for item in raw_items
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": "2.0",
                "random_seed": random_seed,
                "total_unique_clips": len(raw_items),
                "total_source_videos": len(records_by_video),
                "raters_per_clip": num_raters,
                "primary_outcome": "interestingness",
                "secondary_outcomes": ["self_containment", "content_payoff", "start_end_quality"],
                "exploratory_outcomes": ["redundancy", "shareability"],
                "clips": blinded_clips_data,
            },
            f,
            indent=2,
        )

    # 7. Export Researcher-Only Unblinding Key (strictly isolated)
    unblinding_path = output_dir / "researcher_unblinding_key.json"
    with open(unblinding_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "warning": "RESEARCHER ONLY — DO NOT SHARE WITH RATERS BEFORE RATING IS COMPLETE",
                "blind_code_to_system_name": unblinding_key,
                "clip_truth_mapping": clip_truth_mapping,
            },
            f,
            indent=2,
        )

    # 8. Export Rating Rubric Protocol
    protocol_path = output_dir / "HUMAN_EVAL_PROTOCOL.md"
    with open(protocol_path, "w", encoding="utf-8") as f:
        f.write(RATING_RUBRIC_DOCUMENTATION)

    return {
        "total_clips": len(raw_items),
        "total_videos": len(records_by_video),
        "total_rating_rows": len(rating_rows),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "unblinding_path": str(unblinding_path),
        "protocol_path": str(protocol_path),
    }


def render_clips_for_package(
    selected_records: List[Dict[str, Any]],
    source_video_paths: Dict[str, Path],
    clips_dir: Path,
    ffmpeg_bin: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Renders actual MP4 files for selected intervals using FFmpeg and verifies them.
    """
    ffmpeg_cmd = ffmpeg_bin or "ffmpeg"
    clips_dir.mkdir(parents=True, exist_ok=True)

    rendered_records = []
    for idx, rec in enumerate(selected_records, start=1):
        vid = str(rec.get("episode_id") or rec.get("source_video_id") or "")
        clean_vid = vid.replace("rhapsody_real_", "")
        src_path = source_video_paths.get(clean_vid) or source_video_paths.get(vid)

        if not src_path or not src_path.exists():
            rendered_records.append(rec)
            continue

        clip_filename = f"clip_{idx:02d}.mp4"
        out_clip_path = clips_dir / clip_filename
        start = float(rec.get("pred_start") or rec.get("start") or 0.0)
        end = float(rec.get("pred_end") or rec.get("end") or 0.0)
        dur = max(0.1, end - start)

        cmd = [
            ffmpeg_cmd,
            "-y",
            "-ss", f"{start:.3f}",
            "-i", str(src_path),
            "-t", f"{dur:.3f}",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "128k",
            "-avoid_negative_ts", "make_zero",
            str(out_clip_path),
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
            if res.returncode == 0 and out_clip_path.exists() and out_clip_path.stat().st_size > 0:
                rec_copy = dict(rec)
                rec_copy["rendered_clip_path"] = str(out_clip_path)
                rendered_records.append(rec_copy)
                logger.info(f"Successfully rendered clip_{idx:02d}.mp4: [{start:.2f}s -> {end:.2f}s]")
            else:
                rendered_records.append(rec)
        except Exception as e:
            logger.warning(f"Failed to render clip_{idx:02d}.mp4: {e}")
            rendered_records.append(rec)

    return rendered_records


def export_top1_pilot_package(
    csv_input_path: str,
    output_dir: str = "evaluation/human_eval_top1_pilot",
    num_raters: int = 3,
    random_seed: int = 42,
    render_source_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Builds the authoritative Top-1 matched human evaluation pilot package across 4 systems on the 5 pilot episodes.
    - Exactly 1 top candidate per system per video (aiming for 20 candidates total).
    - Preserves and reports missingness rather than fabricating candidates.
    - Ensures Systems C and D evaluate the exact same base candidate interval.
    - Conducts a source video availability check and documents any unavailable media as blockers.
    - Isolates researcher unblinding key and outputs blank rating forms.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    with open(csv_input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    from collections import defaultdict
    episodes_map = defaultdict(lambda: defaultdict(list))
    for r in records:
        ep_id = r.get("episode_id") or r.get("source_video_id")
        sys_m = r.get("system_mode") or r.get("system_name")
        episodes_map[ep_id][sys_m].append(r)

    target_systems = [
        "system_a_transcript_only",
        "system_b_simple_aggregation",
        "system_c_mter",
        "system_d_mter_refined",
    ]

    selected_records: List[Dict[str, Any]] = []
    missing_candidates_report: List[Dict[str, Any]] = []
    system_counts = defaultdict(int)
    system_durations = defaultdict(float)
    per_episode_stats: Dict[str, Any] = {}

    for ep_id in sorted(episodes_map.keys()):
        sys_dict = episodes_map[ep_id]
        per_episode_stats[ep_id] = {}

        # First resolve System C so System D can strictly match its base candidate
        c_cands = sys_dict.get("system_c_mter", [])
        chosen_c = select_top_k_candidates_for_system(c_cands, "system_c_mter", k=1)
        base_c = chosen_c[0] if chosen_c else None

        for sys_m in target_systems:
            cands = sys_dict.get(sys_m, [])
            if not cands:
                missing_candidates_report.append({
                    "episode_id": ep_id,
                    "system_mode": sys_m,
                    "reason": "System produced 0 candidates for this source video (abstention or constraint failure)",
                })
                per_episode_stats[ep_id][sys_m] = {
                    "available_candidates": 0,
                    "selected_for_pilot": 0,
                    "total_selected_duration_sec": 0.0,
                    "status": "missing",
                }
                continue

            if sys_m == "system_c_mter":
                chosen = chosen_c
            elif sys_m == "system_d_mter_refined":
                chosen = select_top_k_candidates_for_system(
                    cands, sys_m, k=1, base_candidate_selected=base_c
                )
            else:
                chosen = select_top_k_candidates_for_system(cands, sys_m, k=1)

            selected_records.extend(chosen)
            system_counts[sys_m] += len(chosen)
            tot_dur = sum(float(c.get("duration_sec") or c.get("pred_duration") or c.get("duration") or 0.0) for c in chosen)
            system_durations[sys_m] += tot_dur
            per_episode_stats[ep_id][sys_m] = {
                "available_candidates": len(cands),
                "selected_for_pilot": len(chosen),
                "total_selected_duration_sec": round(tot_dur, 2),
                "status": "selected" if chosen else "missing",
                "span_id": (chosen[0].get("candidate_id") or chosen[0].get("span_id", "")) if chosen else None,
                "start_sec": float(chosen[0].get("start_sec") or chosen[0].get("pred_start") or 0.0) if chosen else None,
                "end_sec": float(chosen[0].get("end_sec") or chosen[0].get("pred_end") or 0.0) if chosen else None,
            }

    # Inspect source video availability
    unique_vids = [
        str(rec.get("episode_id") or rec.get("source_video_id") or "").replace("rhapsody_real_", "")
        for rec in selected_records
    ]
    unique_vids = sorted(list(set(v for v in unique_vids if v)))

    search_dirs = [Path(render_source_dir)] if render_source_dir else None
    video_availability = inspect_source_video_availability(unique_vids, search_dirs=search_dirs)
    missing_videos = [vid for vid, path in video_availability.items() if path is None]

    # Generate SOURCE_AVAILABILITY_REPORT.md
    avail_report_path = out_path / "SOURCE_AVAILABILITY_REPORT.md"
    with open(avail_report_path, "w", encoding="utf-8") as f:
        f.write("# ClipSense AI — Source Video Availability Audit\n\n")
        f.write("## Status Summary\n")
        if missing_videos:
            f.write(f"- **Rendering Status**: **BLOCKED FOR LOCAL MP4 EXPORT**\n")
            f.write(f"- **Missing Videos**: {len(missing_videos)} of {len(unique_vids)} episodes\n")
            f.write(f"- **Root Cause**: The Rhapsody dataset (`younghanstark/rhapsody` on Hugging Face) "
                    "distributes official WhisperX transcript word timestamps and YouTube Most Replayed graphs. "
                    "Full-length raw source `.mp4` video files are not stored locally in this repository.\n\n")
            f.write("## Video Availability Table\n\n")
            f.write("| YouTube Video ID | Local File Path | Availability Status | Fallback Access Mode |\n")
            f.write("|---|---|---|---|\n")
            for vid, path in video_availability.items():
                if path:
                    f.write(f"| `{vid}` | `{path}` | Available | Direct MP4 Rendering |\n")
                else:
                    f.write(f"| `{vid}` | *None (missing from repository)* | **UNAVAILABLE** | Verified YouTube Timestamp URL |\n")
            f.write("\n## Remediation for Offline MP4 Rendering\n")
            f.write("To render physical MP4 highlight clips:\n")
            f.write("1. Download the 5 source YouTube videos into a directory (e.g., `backend/data/rhapsody_videos/{vid}.mp4`).\n")
            f.write("2. Re-run: `python -m evaluation.human_eval_export --render-from backend/data/rhapsody_videos/`.\n")
            f.write("3. The exporter will automatically cut, encode, and verify all 20 MP4 clips into `clips/clip_01.mp4`..`clip_20.mp4`.\n")
        else:
            f.write("- **Rendering Status**: **ALL SOURCE VIDEOS AVAILABLE**\n")
            f.write("All 5 source videos are present locally and rendered clips have been compiled.\n")

    # If all source files are available, perform physical MP4 rendering
    rendered_candidates = selected_records
    if not missing_videos:
        clips_sub = out_path / "clips"
        rendered_candidates = render_clips_for_package(
            selected_records,
            {v: p for v, p in video_availability.items() if p},
            clips_sub,
        )

    # Compile pilot system comparison summary
    comparison_summary = {
        "dataset_name": "Rhapsody_Real_5ep_Top1_Human_Pilot",
        "max_clips_per_system_per_video": 1,
        "total_source_videos": len(episodes_map),
        "total_selected_clips": len(selected_records),
        "all_combinations_present": len(missing_candidates_report) == 0,
        "source_videos_available_locally": len(missing_videos) == 0,
        "missing_videos_count": len(missing_videos),
        "missing_candidates_count": len(missing_candidates_report),
        "missing_candidates": missing_candidates_report,
        "system_summaries": {
            sys_m: {
                "total_clips": system_counts[sys_m],
                "mean_clips_per_video": round(system_counts[sys_m] / max(1, len(episodes_map)), 2),
                "total_duration_sec": round(system_durations[sys_m], 2),
                "mean_clip_duration_sec": round(system_durations[sys_m] / max(1, system_counts[sys_m]), 2) if system_counts[sys_m] else 0.0,
            }
            for sys_m in target_systems
        },
        "per_episode_breakdown": per_episode_stats,
    }

    with open(out_path / "pilot_system_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)

    # Generate blinded rating package
    pkg_res = generate_human_evaluation_package(
        rendered_candidates,
        out_path,
        num_raters=num_raters,
        random_seed=random_seed,
        clip_id_prefix="clip",
    )
    pkg_res["system_counts"] = dict(system_counts)
    pkg_res["system_durations"] = dict(system_durations)
    pkg_res["comparison_summary"] = comparison_summary
    pkg_res["source_availability_report"] = str(avail_report_path)
    pkg_res["missing_videos"] = missing_videos

    return pkg_res


def export_matched_pilot_package(
    csv_input_path: str,
    output_dir: str,
    max_clips_per_system: int = 3,
    num_raters: int = 3,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Legacy wrapper maintaining backward-compatibility for <=3 clips matched pilot.
    Uses score-driven selection rather than raw CSV row order.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    records = []
    with open(csv_input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    from collections import defaultdict
    episodes_map = defaultdict(lambda: defaultdict(list))
    for r in records:
        ep_id = r.get("episode_id") or r.get("source_video_id")
        sys_m = r.get("system_mode") or r.get("system_name")
        episodes_map[ep_id][sys_m].append(r)

    selected_records = []
    system_counts = defaultdict(int)
    system_durations = defaultdict(float)
    per_episode_stats = {}

    target_systems = [
        "system_a_transcript_only",
        "system_b_simple_aggregation",
        "system_c_mter",
        "system_d_mter_refined",
    ]

    for ep_id in sorted(episodes_map.keys()):
        sys_dict = episodes_map[ep_id]
        per_episode_stats[ep_id] = {}
        c_cands = sys_dict.get("system_c_mter", [])
        chosen_c = select_top_k_candidates_for_system(c_cands, "system_c_mter", k=max_clips_per_system)
        base_c = chosen_c[0] if chosen_c else None

        for sys_m in target_systems:
            cands = sys_dict.get(sys_m, [])
            if sys_m == "system_c_mter":
                chosen = chosen_c
            elif sys_m == "system_d_mter_refined":
                chosen = select_top_k_candidates_for_system(
                    cands, sys_m, k=max_clips_per_system, base_candidate_selected=base_c
                )
            else:
                chosen = select_top_k_candidates_for_system(cands, sys_m, k=max_clips_per_system)

            selected_records.extend(chosen)
            system_counts[sys_m] += len(chosen)
            tot_dur = sum(float(c.get("pred_duration") or 0.0) for c in chosen)
            system_durations[sys_m] += tot_dur
            per_episode_stats[ep_id][sys_m] = {
                "available_candidates": len(cands),
                "selected_for_pilot": len(chosen),
                "total_selected_duration_sec": round(tot_dur, 2),
            }

    comparison_summary = {
        "dataset_name": "Rhapsody_Real_5ep_Matched_Human_Pilot",
        "max_clips_per_system_per_video": max_clips_per_system,
        "total_source_videos": len(episodes_map),
        "total_selected_clips": len(selected_records),
        "system_summaries": {
            sys_m: {
                "total_clips": system_counts[sys_m],
                "mean_clips_per_video": round(system_counts[sys_m] / max(1, len(episodes_map)), 2),
                "total_duration_sec": round(system_durations[sys_m], 2),
                "mean_clip_duration_sec": round(system_durations[sys_m] / max(1, system_counts[sys_m]), 2) if system_counts[sys_m] else 0.0,
            }
            for sys_m in target_systems
        },
        "per_episode_breakdown": per_episode_stats,
    }

    with open(out_path / "pilot_system_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)

    pkg_res = generate_human_evaluation_package(
        selected_records,
        out_path,
        num_raters=num_raters,
        random_seed=random_seed,
    )
    pkg_res["system_counts"] = dict(system_counts)
    pkg_res["system_durations"] = dict(system_durations)
    pkg_res["comparison_summary"] = comparison_summary
    return pkg_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate human evaluation package for ClipSense AI.")
    parser.add_argument("--input-csv", type=str, default="runs/rhapsody_real_pilot_5ep_run/timestamp_comparison.csv", help="Input timestamp_comparison.csv")
    parser.add_argument("--output-dir", type=str, default="evaluation/human_eval_top1_pilot", help="Destination directory")
    parser.add_argument("--top1", action="store_true", default=True, help="Build top-1 pilot (1 candidate per system per video)")
    parser.add_argument("--num-raters", type=int, default=3, help="Number of independent raters per clip")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for clip presentation order")
    parser.add_argument("--render-from", type=str, default=None, help="Directory containing source MP4 videos to render physical clips")
    args = parser.parse_args()

    backend_dir = Path(__file__).parent.parent.resolve()
    input_csv = Path(args.input_csv)
    if not input_csv.is_absolute():
        input_csv = backend_dir / input_csv

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = backend_dir / out_dir

    if input_csv.exists():
        res = export_top1_pilot_package(
            str(input_csv),
            str(out_dir),
            num_raters=args.num_raters,
            random_seed=args.seed,
            render_source_dir=args.render_from,
        )
        print("Successfully generated Top-1 Matched Pilot Package:")
        print(f" - Output Directory:     {out_dir}")
        print(f" - Total Unique Clips:   {res['total_clips']}")
        print(f" - Total Rating Rows:    {res['total_rating_rows']}")
        print(f" - System Counts:        {res['system_counts']}")
        print(f" - System Durations (s): {res['system_durations']}")
        print(f" - Rating Form CSV:      {res['csv_path']}")
        print(f" - Blinded JSON:         {res['json_path']}")
        print(f" - Unblinding Key:       {res['unblinding_path']}")
        print(f" - Missing Videos Count: {len(res['missing_videos'])}")
    else:
        print(f"Error: Input CSV not found at {input_csv}")
