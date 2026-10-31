"""
ClipSense AI — Timestamp Comparison and Visual Timeline Generator
Task 2 Deliverable: Standalone diagnostic comparing continuous clip intervals
against Rhapsody 100-segment replay engagement curves.

Important Scientific Caveat:
Rhapsody annotations are viewer replay engagement curves mapped to 100 uniform temporal buckets,
NOT precise human start/end highlight boundaries. Evaluating candidate start/end differences
against coarse bins (e.g. 11s - 28s wide) serves as a macro-temporal localization diagnostic,
not sub-second human cut-point validation.
"""

import csv
import json
import logging
import math
import os
import pathlib
import sys
from typing import Any, Dict, List, Optional

# Ensure backend root is in sys.path
backend_root = str(pathlib.Path(__file__).parent.parent)
if backend_root not in sys.path:
    sys.path.insert(0, backend_root)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("timeline_visualizer")


def generate_timestamp_comparison(
    manifest_path: str,
    eval_results_path: str,
    output_dir: str,
    gt_threshold: float = 0.5,
    max_n_gt: int = 20,
) -> Dict[str, Any]:
    """
    Parses manifest ground truth and evaluation results, producing:
    1. timestamp_comparison.json
    2. timestamp_comparison.csv
    3. timeline_dashboard.html
    """
    manifest_p = pathlib.Path(manifest_path).resolve()
    eval_p = pathlib.Path(eval_results_path).resolve()
    out_p = pathlib.Path(output_dir).resolve()
    out_p.mkdir(parents=True, exist_ok=True)

    if not manifest_p.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_p}")
    if not eval_p.exists():
        raise FileNotFoundError(f"Evaluation results not found: {eval_p}")

    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    with open(eval_p, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    # Index sample evaluation results by (sample_id, mode_name)
    sample_results_map: Dict[str, Dict[str, Any]] = {}
    for sr in eval_data.get("sample_results", []):
        sid = sr.get("sample_id")
        mode = sr.get("mode_name")
        if sid not in sample_results_map:
            sample_results_map[sid] = {}
        sample_results_map[sid][mode] = sr

    episodes_summary = []
    csv_rows = []

    for sample in manifest_data.get("samples", []):
        sid = sample.get("sample_id")
        dur = float(sample.get("duration_sec", 0.0))
        t_bin = dur / 100.0
        meta = sample.get("metadata", {})
        vid = meta.get("youtube_vid", sid)
        title = meta.get("title", vid)
        raw_scores = sample.get("replay_scores", [])

        # Apply official bias correction
        scores_corrected = list(raw_scores)
        if len(scores_corrected) > 1:
            scores_corrected[0] = scores_corrected[1]

        # Extract positive ground truth bins strictly > threshold
        pos_candidates = [
            (idx, float(s))
            for idx, s in enumerate(scores_corrected)
            if float(s) > gt_threshold
        ]
        pos_candidates.sort(key=lambda x: (-x[1], x[0]))
        selected_gt = pos_candidates[:max_n_gt]
        selected_gt.sort(key=lambda x: x[0])  # chronological

        benchmark_bins = []
        for idx, s in selected_gt:
            b_start = round(idx * t_bin, 3)
            b_end = round(min(dur, (idx + 1) * t_bin), 3)
            benchmark_bins.append({
                "bin_index": idx,
                "replay_score": round(s, 4),
                "start_time": b_start,
                "end_time": b_end,
                "duration": round(b_end - b_start, 3),
            })

        # Derived contiguous regions (for timeline visualization only, not official benchmark ground truth)
        merged_regions = []
        if benchmark_bins:
            curr_start = benchmark_bins[0]["start_time"]
            curr_end = benchmark_bins[0]["end_time"]
            curr_bins = [benchmark_bins[0]["bin_index"]]
            max_score = benchmark_bins[0]["replay_score"]

            for b in benchmark_bins[1:]:
                if b["bin_index"] == curr_bins[-1] + 1:
                    curr_end = b["end_time"]
                    curr_bins.append(b["bin_index"])
                    max_score = max(max_score, b["replay_score"])
                else:
                    merged_regions.append({
                        "region_id": f"merged_{curr_bins[0]}_{curr_bins[-1]}",
                        "start_time": curr_start,
                        "end_time": curr_end,
                        "duration": round(curr_end - curr_start, 3),
                        "bin_indices": curr_bins,
                        "max_replay_score": max_score,
                    })
                    curr_start = b["start_time"]
                    curr_end = b["end_time"]
                    curr_bins = [b["bin_index"]]
                    max_score = b["replay_score"]
            merged_regions.append({
                "region_id": f"merged_{curr_bins[0]}_{curr_bins[-1]}",
                "start_time": curr_start,
                "end_time": curr_end,
                "duration": round(curr_end - curr_start, 3),
                "bin_indices": curr_bins,
                "max_replay_score": max_score,
            })

        # Process predicted candidates across systems
        modes_order = [
            "system_a_transcript_only",
            "system_b_simple_aggregation",
            "system_c_mter",
            "system_d_mter_refined",
        ]

        systems_output = {}
        for mode in modes_order:
            sr = sample_results_map.get(sid, {}).get(mode, {})
            matches = sr.get("matched_metrics", [])
            matches_by_pred_id = {m["prediction_span_id"]: m for m in matches}

            # 1. Attempt to load full candidate spans from sr or baseline cache or ablation runner
            candidates_loaded = []
            if "predicted_spans" in sr and sr["predicted_spans"]:
                candidates_loaded = sr["predicted_spans"]
            elif "predictions" in sr and sr["predictions"]:
                candidates_loaded = sr["predictions"]

            # 2. Check baseline cache for system_a_transcript_only
            if not candidates_loaded and mode == "system_a_transcript_only":
                possible_caches = [
                    out_p / "baseline_cache" / f"baseline_{sid}.json",
                    eval_p.parent / "baseline_cache" / f"baseline_{sid}.json",
                ]
                for b_cache in possible_caches:
                    if b_cache.exists():
                        try:
                            with open(b_cache, "r", encoding="utf-8") as bf:
                                candidates_loaded = json.load(bf)
                            break
                        except Exception:
                            pass
                if not candidates_loaded:
                    alt_caches = list(out_p.parent.glob(f"**/baseline_{sid}.json"))
                    if alt_caches:
                        try:
                            with open(alt_caches[0], "r", encoding="utf-8") as bf:
                                candidates_loaded = json.load(bf)
                        except Exception:
                            pass

            # 3. If still empty, reconstruct candidates deterministically via AblationRunner
            run_dir = sample.get("run_dir")
            if not candidates_loaded and run_dir:
                s_run_path = pathlib.Path(run_dir)
                if not s_run_path.is_absolute():
                    s_run_path = (pathlib.Path(backend_root) / s_run_path).resolve()
                if s_run_path.exists():
                    try:
                        from evaluation.ablation_runner import AblationRunner
                        runner = AblationRunner()
                        spans = runner.run_mode(
                            mode=mode,
                            run_dir=str(s_run_path),
                            video_duration=dur,
                            sample_id=sid,
                            cache_dir=str(out_p / "baseline_cache"),
                        )
                        candidates_loaded = [
                            {
                                "span_id": s.span_id,
                                "start_time": s.start_time,
                                "end_time": s.end_time,
                                "duration": s.duration,
                                "confidence": s.confidence,
                            }
                            for s in spans
                        ]
                    except Exception as ex:
                        logger.debug(f"AblationRunner fallback skipped for {mode} on {sid}: {ex}")

            system_candidates = []
            seen_cand_ids = set()

            # First populate candidates with known start/end timestamps
            for c in candidates_loaded:
                c_id = c.get("span_id") if isinstance(c, dict) else getattr(c, "span_id", "")
                c_s = c.get("start_time") if isinstance(c, dict) else getattr(c, "start_time", 0.0)
                c_e = c.get("end_time") if isinstance(c, dict) else getattr(c, "end_time", 0.0)
                c_dur = c.get("duration") if isinstance(c, dict) else getattr(c, "duration", (c_e - c_s))
                c_conf = c.get("confidence") if isinstance(c, dict) else getattr(c, "confidence", None)
                if c_conf is None:
                    c_conf = c.get("score") if isinstance(c, dict) else getattr(c, "score", None)
                conf_val = round(float(c_conf), 4) if c_conf is not None else None
                seen_cand_ids.add(c_id)

                if c_id in matches_by_pred_id:
                    match = matches_by_pred_id[c_id]
                    system_candidates.append({
                        "span_id": c_id,
                        "start_time": match["pred_start"],
                        "end_time": match["pred_end"],
                        "duration": match["pred_duration"],
                        "inference_score": conf_val,
                        "matched": True,
                        "matched_reference_id": match["reference_span_id"],
                        "benchmark_start": match["ref_start"],
                        "benchmark_end": match["ref_end"],
                        "start_error_sec": match["start_error_sec"],
                        "end_error_sec": match["end_error_sec"],
                        "temporal_iou": match["temporal_iou"],
                    })
                    csv_rows.append({
                        "episode_id": sid,
                        "youtube_vid": vid,
                        "system_mode": mode,
                        "span_id": c_id,
                        "pred_start": match["pred_start"],
                        "pred_end": match["pred_end"],
                        "pred_duration": match["pred_duration"],
                        "inference_score": conf_val,
                        "matched": True,
                        "benchmark_start": match["ref_start"],
                        "benchmark_end": match["ref_end"],
                        "start_error_sec": match["start_error_sec"],
                        "end_error_sec": match["end_error_sec"],
                        "temporal_iou": match["temporal_iou"],
                    })
                else:
                    system_candidates.append({
                        "span_id": c_id,
                        "start_time": c_s,
                        "end_time": c_e,
                        "duration": c_dur,
                        "inference_score": conf_val,
                        "matched": False,
                        "matched_reference_id": None,
                        "benchmark_start": None,
                        "benchmark_end": None,
                        "start_error_sec": None,
                        "end_error_sec": None,
                        "temporal_iou": 0.0,
                    })
                    csv_rows.append({
                        "episode_id": sid,
                        "youtube_vid": vid,
                        "system_mode": mode,
                        "span_id": c_id,
                        "pred_start": c_s,
                        "pred_end": c_e,
                        "pred_duration": c_dur,
                        "inference_score": conf_val,
                        "matched": False,
                        "benchmark_start": None,
                        "benchmark_end": None,
                        "start_error_sec": None,
                        "end_error_sec": None,
                        "temporal_iou": 0.0,
                    })

            # For any matches not yet included in system_candidates
            for match in matches:
                p_id = match["prediction_span_id"]
                if p_id in seen_cand_ids:
                    continue
                seen_cand_ids.add(p_id)
                system_candidates.append({
                    "span_id": p_id,
                    "start_time": match["pred_start"],
                    "end_time": match["pred_end"],
                    "duration": match["pred_duration"],
                    "inference_score": None,
                    "matched": True,
                    "matched_reference_id": match["reference_span_id"],
                    "benchmark_start": match["ref_start"],
                    "benchmark_end": match["ref_end"],
                    "start_error_sec": match["start_error_sec"],
                    "end_error_sec": match["end_error_sec"],
                    "temporal_iou": match["temporal_iou"],
                })
                csv_rows.append({
                    "episode_id": sid,
                    "youtube_vid": vid,
                    "system_mode": mode,
                    "span_id": p_id,
                    "pred_start": match["pred_start"],
                    "pred_end": match["pred_end"],
                    "pred_duration": match["pred_duration"],
                    "inference_score": None,
                    "matched": True,
                    "benchmark_start": match["ref_start"],
                    "benchmark_end": match["ref_end"],
                    "start_error_sec": match["start_error_sec"],
                    "end_error_sec": match["end_error_sec"],
                    "temporal_iou": match["temporal_iou"],
                })

            # For any unmatched_prediction_ids not yet in seen_cand_ids
            unmatched_p_ids = sr.get("unmatched_prediction_ids", [])
            for up_id in unmatched_p_ids:
                if up_id in seen_cand_ids:
                    continue
                seen_cand_ids.add(up_id)
                system_candidates.append({
                    "span_id": up_id,
                    "start_time": None,
                    "end_time": None,
                    "duration": None,
                    "inference_score": None,
                    "matched": False,
                    "matched_reference_id": None,
                    "benchmark_start": None,
                    "benchmark_end": None,
                    "start_error_sec": None,
                    "end_error_sec": None,
                    "temporal_iou": 0.0,
                })
                csv_rows.append({
                    "episode_id": sid,
                    "youtube_vid": vid,
                    "system_mode": mode,
                    "span_id": up_id,
                    "pred_start": None,
                    "pred_end": None,
                    "pred_duration": None,
                    "inference_score": None,
                    "matched": False,
                    "benchmark_start": None,
                    "benchmark_end": None,
                    "start_error_sec": None,
                    "end_error_sec": None,
                    "temporal_iou": 0.0,
                })

            systems_output[mode] = {
                "num_predictions": sr.get("num_predictions", 0),
                "num_matched": sr.get("num_matched", 0),
                "candidates": system_candidates,
            }

        episodes_summary.append({
            "sample_id": sid,
            "youtube_vid": vid,
            "title": title,
            "duration_sec": dur,
            "bin_duration_sec": round(t_bin, 3),
            "replay_scores_corrected": [round(float(x), 4) for x in scores_corrected],
            "benchmark_positive_bins": benchmark_bins,
            "visualization_merged_regions": merged_regions,
            "systems": systems_output,
        })

    # Save JSON report
    json_path = out_p / "timestamp_comparison.json"
    comparison_obj = {
        "dataset_name": manifest_data.get("dataset_name", "Rhapsody_Pilot"),
        "total_episodes": len(episodes_summary),
        "protocol_note": (
            "Benchmark positive bins represent YouTube Most Replayed crowd engagement curves "
            "(gt_threshold > 0.50, max_n_gt = 20) with segment 0 bias correction (s_0 = s_1). "
            "Continuous clip candidates are projected to bins via maximum confidence-weighted overlap. "
            "Start/end deltas are reported against coarse bins (11s - 28s width), serving as macro-temporal "
            "localization diagnostics rather than sub-second human cut-point validation."
        ),
        "episodes": episodes_summary,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_obj, f, indent=2)

    # Save CSV report
    csv_path = out_p / "timestamp_comparison.csv"
    if csv_rows:
        fieldnames = [
            "episode_id", "youtube_vid", "system_mode", "span_id",
            "pred_start", "pred_end", "pred_duration", "inference_score", "matched",
            "benchmark_start", "benchmark_end", "start_error_sec",
            "end_error_sec", "temporal_iou"
        ]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    # Generate HTML Dashboard
    html_path = out_p / "timeline_dashboard.html"
    generate_html_dashboard(comparison_obj, str(html_path))

    logger.info(f"Generated timestamp comparison JSON at: {json_path}")
    logger.info(f"Generated timestamp comparison CSV at: {csv_path}")
    logger.info(f"Generated visual timeline HTML dashboard at: {html_path}")

    return comparison_obj


def generate_html_dashboard(data: Dict[str, Any], output_path: str):
    """Renders a self-contained HTML timeline dashboard."""
    episodes = data.get("episodes", [])
    episodes_json = json.dumps(episodes)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ClipSense AI × Rhapsody — Timestamp & Timeline Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg-main: #0B0F19;
    --bg-card: #111827;
    --bg-card-sub: #1F2937;
    --border-color: #374151;
    --text-primary: #F9FAFB;
    --text-secondary: #9CA3AF;
    --text-muted: #6B7280;
    --accent-amber: #F59E0B;
    --accent-cyan: #06B6D4;
    --accent-indigo: #8B5CF6;
    --accent-emerald: #10B981;
    --accent-rose: #F43F5E;
    --radius-lg: 12px;
    --radius-md: 8px;
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.4), 0 2px 4px -1px rgba(0, 0, 0, 0.2);
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background-color: var(--bg-main);
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
    line-height: 1.5;
    padding: 2rem;
  }}

  .container {{
    max-width: 1400px;
    margin: 0 auto;
  }}

  header {{
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border-color);
  }}

  h1 {{
    font-size: 1.875rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }}

  .badge {{
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.25rem 0.625rem;
    border-radius: 9999px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .badge-research {{ background: rgba(139, 92, 246, 0.2); color: #C4B5FD; border: 1px solid #8B5CF6; }}
  .badge-benchmark {{ background: rgba(245, 158, 11, 0.2); color: #FCD34D; border: 1px solid #F59E0B; }}

  .alert-box {{
    background: rgba(31, 41, 55, 0.7);
    border: 1px solid var(--border-color);
    border-left: 4px solid var(--accent-amber);
    padding: 1rem 1.25rem;
    border-radius: var(--radius-md);
    margin: 1.5rem 0;
    font-size: 0.875rem;
    color: #D1D5DB;
  }}

  .controls-bar {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    background: var(--bg-card);
    padding: 1rem 1.5rem;
    border-radius: var(--radius-lg);
    border: 1px solid var(--border-color);
    margin-bottom: 2rem;
  }}

  select {{
    background: var(--bg-card-sub);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
    padding: 0.5rem 1rem;
    border-radius: var(--radius-md);
    font-size: 0.9375rem;
    font-family: inherit;
    cursor: pointer;
    min-width: 320px;
  }}

  .meta-stats {{
    display: flex;
    gap: 1.5rem;
    font-size: 0.875rem;
    color: var(--text-secondary);
  }}
  .meta-stats span strong {{ color: var(--text-primary); }}

  .card {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin-bottom: 2rem;
    box-shadow: var(--shadow-md);
  }}

  .card-title {{
    font-size: 1.125rem;
    font-weight: 700;
    margin-bottom: 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}

  /* Timeline Canvas Styling */
  .timeline-container {{
    position: relative;
    width: 100%;
    margin-top: 1rem;
    background: #0D1117;
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    padding: 1rem;
    overflow-x: auto;
  }}

  svg.timeline-svg {{
    display: block;
    width: 100%;
    min-width: 800px;
    height: 320px;
  }}

  .track-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    fill: var(--text-secondary);
  }}

  /* Comparison Table */
  table.data-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
    margin-top: 1rem;
  }}

  th, td {{
    padding: 0.75rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border-color);
  }}

  th {{
    background: var(--bg-card-sub);
    color: var(--text-secondary);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
  }}

  td.mono {{ font-family: 'JetBrains Mono', monospace; }}

  .tag {{
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
  }}
  .tag-sys-a {{ background: rgba(6, 182, 212, 0.2); color: #67E8F9; border: 1px solid #06B6D4; }}
  .tag-sys-b {{ background: rgba(139, 92, 246, 0.2); color: #C4B5FD; border: 1px solid #8B5CF6; }}
  .tag-sys-c {{ background: rgba(16, 185, 129, 0.2); color: #6EE7B7; border: 1px solid #10B981; }}
  .tag-sys-d {{ background: rgba(244, 63, 94, 0.2); color: #FDA4AF; border: 1px solid #F43F5E; }}
  .tag-gt {{ background: rgba(245, 158, 11, 0.2); color: #FCD34D; border: 1px solid #F59E0B; }}

  .legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 1.25rem;
    font-size: 0.8125rem;
    margin-top: 1rem;
    padding-top: 0.75rem;
    border-top: 1px solid var(--border-color);
  }}

  .legend-item {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }}

  .legend-color {{
    width: 14px;
    height: 14px;
    border-radius: 3px;
  }}
</style>
</head>
<body>

<div class="container">
  <header>
    <h1>
      ClipSense AI × Rhapsody
      <span class="badge badge-research">Research Diagnostic</span>
      <span class="badge badge-benchmark">Test Split (5 Episodes)</span>
    </h1>
    <p style="color: var(--text-secondary); margin-top: 0.5rem;">
      Continuous highlight candidates projected against official YouTube Most Replayed engagement curves (Park et al., COLM 2025).
    </p>
  </header>

  <div class="alert-box">
    <strong>Scientific Protocol Notice:</strong>
    Rhapsody benchmark ground truth represents crowd playback intensity telemetry sampled into 100 uniform temporal buckets (11s – 28s width),
    <strong>not</strong> manually clicked human sentence cuts. Start and end differences reported below serve as macro-temporal localization diagnostics
    (confirming whether ClipSense identified the same climax as the audience), not sub-second human cut-point validation.
  </div>

  <div class="controls-bar">
    <div>
      <label for="episodeSelect" style="font-weight: 600; margin-right: 0.5rem;">Select Episode:</label>
      <select id="episodeSelect" onchange="renderSelectedEpisode()"></select>
    </div>
    <div class="meta-stats" id="episodeMeta"></div>
  </div>

  <div class="card">
    <div class="card-title">
      <span>Macro-Temporal Alignment Timeline</span>
      <span style="font-size: 0.8125rem; font-weight: 400; color: var(--text-muted);" id="timelineScaleInfo"></span>
    </div>
    <div class="timeline-container">
      <svg id="timelineSvg" class="timeline-svg" viewBox="0 0 1000 320" preserveAspectRatio="none"></svg>
    </div>
    <div class="legend">
      <div class="legend-item"><div class="legend-color" style="background: var(--accent-amber);"></div> Benchmark Ground-Truth Peak (YouTube Replay &gt; 0.50)</div>
      <div class="legend-item"><div class="legend-color" style="background: var(--accent-cyan);"></div> System A: Transcript-Only Baseline</div>
      <div class="legend-item"><div class="legend-color" style="background: var(--accent-indigo);"></div> System B: Simple Aggregation</div>
      <div class="legend-item"><div class="legend-color" style="background: var(--accent-emerald);"></div> System C: MTER (Arbitrated)</div>
      <div class="legend-item"><div class="legend-color" style="background: var(--accent-rose);"></div> System D: MTER + Refined Boundaries</div>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Detailed Timestamp Distances (Matched &amp; Localized Clips)</div>
    <div style="overflow-x: auto;">
      <table class="data-table" id="comparisonTable">
        <thead>
          <tr>
            <th>System</th>
            <th>Candidate Span ID</th>
            <th>Predicted Interval [s]</th>
            <th>Duration</th>
            <th>Benchmark Target [s]</th>
            <th>Start Error</th>
            <th>End Error</th>
            <th>Temporal IoU</th>
            <th>Match Status</th>
          </tr>
        </thead>
        <tbody id="tableBody"></tbody>
      </table>
    </div>
  </div>
</div>

<script>
  const episodesData = {episodes_json};

  function init() {{
    const select = document.getElementById('episodeSelect');
    select.innerHTML = '';
    episodesData.forEach((ep, idx) => {{
      const opt = document.createElement('option');
      opt.value = idx;
      opt.textContent = `${{ep.youtube_vid}} — ${{ep.title.substring(0, 50)}} (${{ep.duration_sec.toFixed(0)}}s)`;
      select.appendChild(opt);
    }});
    renderSelectedEpisode();
  }}

  function renderSelectedEpisode() {{
    const idx = parseInt(document.getElementById('episodeSelect').value, 10);
    const ep = episodesData[idx];
    if (!ep) return;

    // Render Metadata
    const metaDiv = document.getElementById('episodeMeta');
    metaDiv.innerHTML = `
      <span>Duration: <strong>${{ep.duration_sec.toFixed(1)}}s (${{(ep.duration_sec / 60).toFixed(1)}}m)</strong></span>
      <span>Bin Width (1%): <strong>${{ep.bin_duration_sec.toFixed(2)}}s</strong></span>
      <span>Benchmark Peaks: <strong>${{ep.benchmark_positive_bins.length}} bins</strong></span>
    `;

    document.getElementById('timelineScaleInfo').textContent = `Total Time Axis: 0.0s to ${{ep.duration_sec.toFixed(1)}}s`;

    // Render Timeline SVG
    renderSvgTimeline(ep);

    // Render Table
    renderComparisonTable(ep);
  }}

  function renderSvgTimeline(ep) {{
    const svg = document.getElementById('timelineSvg');
    const dur = ep.duration_sec;
    const scores = ep.replay_scores_corrected;
    const w = 1000;
    const padX = 140; // track label space
    const plotW = w - padX - 20;

    const timeToX = (t) => padX + (Math.max(0, Math.min(dur, t)) / dur) * plotW;

    let svgHtml = '';

    // Track 1: Replay Curve (Y: 20 to 110)
    svgHtml += `<text x="10" y="65" class="track-label">YouTube Replay</text>`;
    svgHtml += `<line x1="${{padX}}" y1="65" x2="${{w-20}}" y2="65" stroke="#374151" stroke-dasharray="3,3" />`;
    svgHtml += `<text x="${{w-18}}" y="68" fill="#6B7280" font-size="9">0.50</text>`;

    // Build curve path
    let curvePath = `M ${{timeToX(0)}} 110 `;
    for (let i = 0; i < scores.length; i++) {{
      const t = (i + 0.5) * ep.bin_duration_sec;
      const x = timeToX(t);
      const y = 110 - (scores[i] * 90);
      curvePath += `L ${{x.toFixed(1)}} ${{y.toFixed(1)}} `;
    }}
    curvePath += `L ${{timeToX(dur)}} 110 Z`;
    svgHtml += `<path d="${{curvePath}}" fill="rgba(245, 158, 11, 0.15)" stroke="#F59E0B" stroke-width="1.5" />`;

    // Mark Ground Truth positive bins
    ep.benchmark_positive_bins.forEach(b => {{
      const x1 = timeToX(b.start_time);
      const x2 = timeToX(b.end_time);
      svgHtml += `<rect x="${{x1}}" y="20" width="${{Math.max(2, x2 - x1)}}" height="90" fill="rgba(245, 158, 11, 0.25)" stroke="#F59E0B" stroke-width="1" />`;
    }});

    // Tracks definition
    const tracks = [
      {{ id: 'gt', label: 'Benchmark GT', y: 130, color: '#F59E0B', spans: ep.benchmark_positive_bins }},
      {{ id: 'system_a_transcript_only', label: 'System A (Trans)', y: 165, color: '#06B6D4', spans: ep.systems['system_a_transcript_only']?.candidates || [] }},
      {{ id: 'system_b_simple_aggregation', label: 'System B (Agg)', y: 200, color: '#8B5CF6', spans: ep.systems['system_b_simple_aggregation']?.candidates || [] }},
      {{ id: 'system_c_mter', label: 'System C (MTER)', y: 235, color: '#10B981', spans: ep.systems['system_c_mter']?.candidates || [] }},
      {{ id: 'system_d_mter_refined', label: 'System D (Refined)', y: 270, color: '#F43F5E', spans: ep.systems['system_d_mter_refined']?.candidates || [] }}
    ];

    tracks.forEach(tr => {{
      svgHtml += `<text x="10" y="${{tr.y + 11}}" class="track-label">${{tr.label}}</text>`;
      svgHtml += `<line x1="${{padX}}" y1="${{tr.y + 7}}" x2="${{w-20}}" y2="${{tr.y + 7}}" stroke="#1F2937" stroke-width="1" />`;

      tr.spans.forEach(s => {{
        if (s.start_time !== null && s.end_time !== null) {{
          const x1 = timeToX(s.start_time);
          const x2 = timeToX(s.end_time);
          const barW = Math.max(3, x2 - x1);
          const isMatched = s.matched === true;
          const fillStyle = isMatched ? tr.color : 'transparent';
          const strokeStyle = tr.color;
          const strokeWidth = isMatched ? '1' : '1.5';
          const strokeDash = isMatched ? 'none' : '3,2';
          const opacityVal = isMatched ? '0.90' : '0.65';
          const statusText = isMatched ? `Matched (tIoU: ${{s.temporal_iou !== undefined ? s.temporal_iou.toFixed(3) : ''}})` : 'Unmatched';
          svgHtml += `<rect x="${{x1.toFixed(1)}}" y="${{tr.y}}" width="${{barW.toFixed(1)}}" height="14" rx="2" fill="${{fillStyle}}" stroke="${{strokeStyle}}" stroke-width="${{strokeWidth}}" stroke-dasharray="${{strokeDash}}" opacity="${{opacityVal}}">
            <title>${{s.span_id || 'GT'}}: [${{s.start_time.toFixed(1)}}s -> ${{s.end_time.toFixed(1)}}s] (${{statusText}})</title>
          </rect>`;
        }}
      }});
    }});

    // Time Axis at bottom
    svgHtml += `<line x1="${{padX}}" y1="300" x2="${{w-20}}" y2="300" stroke="#4B5563" stroke-width="1" />`;
    const numTicks = 6;
    for (let i = 0; i <= numTicks; i++) {{
      const t = (dur / numTicks) * i;
      const x = timeToX(t);
      svgHtml += `<line x1="${{x}}" y1="300" x2="${{x}}" y2="305" stroke="#9CA3AF" stroke-width="1" />`;
      svgHtml += `<text x="${{x}}" y="316" fill="#9CA3AF" font-size="9" text-anchor="middle" font-family="JetBrains Mono">${{t.toFixed(0)}}s</text>`;
    }}

    svg.innerHTML = svgHtml;
  }}

  function renderComparisonTable(ep) {{
    const tbody = document.getElementById('tableBody');
    tbody.innerHTML = '';

    const modes = [
      {{ id: 'system_a_transcript_only', tag: 'tag-sys-a', name: 'System A' }},
      {{ id: 'system_b_simple_aggregation', tag: 'tag-sys-b', name: 'System B' }},
      {{ id: 'system_c_mter', tag: 'tag-sys-c', name: 'System C' }},
      {{ id: 'system_d_mter_refined', tag: 'tag-sys-d', name: 'System D' }},
    ];

    let rowCount = 0;

    modes.forEach(m => {{
      const sysData = ep.systems[m.id];
      if (!sysData || !sysData.candidates) return;

      sysData.candidates.forEach(c => {{
        if (c.start_time === null) return; // skip completely unpopulated

        rowCount++;
        const tr = document.createElement('tr');
        const predInterval = c.start_time !== null ? `[${{c.start_time.toFixed(1)}}s -> ${{c.end_time.toFixed(1)}}s]` : '—';
        const predDur = c.duration !== null ? `${{c.duration.toFixed(1)}}s` : '—';
        const bmInterval = c.benchmark_start !== null ? `[${{c.benchmark_start.toFixed(1)}}s -> ${{c.benchmark_end.toFixed(1)}}s]` : 'Unmatched';
        const startErr = c.start_error_sec !== null ? `${{c.start_error_sec.toFixed(2)}}s` : 'N/A';
        const endErr = c.end_error_sec !== null ? `${{c.end_error_sec.toFixed(2)}}s` : 'N/A';
        const tiou = c.temporal_iou !== undefined && c.matched ? c.temporal_iou.toFixed(3) : '0.000';
        const statusBadge = c.matched
          ? `<span style="color: #10B981; font-weight: 600;">✓ Matched Peak</span>`
          : `<span style="color: #94A3B8; font-style: italic;">Unmatched Candidate</span>`;

        tr.innerHTML = `
          <td><span class="tag ${{m.tag}}">${{m.name}}</span></td>
          <td class="mono">${{c.span_id}}</td>
          <td class="mono">${{predInterval}}</td>
          <td class="mono">${{predDur}}</td>
          <td class="mono">${{bmInterval}}</td>
          <td class="mono" style="color: ${{c.start_error_sec !== null && c.start_error_sec < 10 ? '#6EE7B7' : '#F9FAFB'}};">${{startErr}}</td>
          <td class="mono" style="color: ${{c.end_error_sec !== null && c.end_error_sec < 10 ? '#6EE7B7' : '#F9FAFB'}};">${{endErr}}</td>
          <td class="mono" style="font-weight: 700;">${{tiou}}</td>
          <td>${{statusBadge}}</td>
        `;
        tbody.appendChild(tr);
      }});
    }});

    if (rowCount === 0) {{
      const tr = document.createElement('tr');
      tr.innerHTML = `<td colspan="9" style="text-align: center; color: var(--text-muted); padding: 2rem;">No localized candidates found for this episode.</td>`;
      tbody.appendChild(tr);
    }}
  }}

  window.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ClipSense AI Timeline Dashboard Visualizer")
    parser.add_argument("--manifest", type=str, default=None, help="Path to benchmark dataset JSON manifest")
    parser.add_argument("--eval-results", type=str, default=None, help="Path to eval_results.json")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory containing eval_results.json")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to write comparison and dashboard")
    parser.add_argument("--gt-threshold", type=float, default=0.5, help="Ground truth replay score threshold")
    parser.add_argument("--max-n-gt", type=int, default=20, help="Maximum number of ground truth bins")
    args = parser.parse_args()

    base_dir = pathlib.Path(__file__).parent.parent
    run_dir = pathlib.Path(args.run_dir) if args.run_dir else None

    if args.output_dir:
        out_dir = pathlib.Path(args.output_dir)
    elif run_dir:
        out_dir = run_dir
    else:
        out_dir = base_dir / "runs" / "rhapsody_real_pilot_5ep_run"

    if args.eval_results:
        results_path = pathlib.Path(args.eval_results)
    elif run_dir and (run_dir / "eval_results.json").exists():
        results_path = run_dir / "eval_results.json"
    else:
        results_path = out_dir / "eval_results.json"

    if args.manifest:
        manifest_path = pathlib.Path(args.manifest)
    else:
        manifest_path = base_dir / "evaluation" / "datasets" / "rhapsody_real_test_5ep.json"
        if not manifest_path.exists():
            manifest_path = base_dir / "evaluation" / "datasets" / "rhapsody_real_test_1ep.json"

    generate_timestamp_comparison(
        manifest_path=str(manifest_path),
        eval_results_path=str(results_path),
        output_dir=str(out_dir),
        gt_threshold=args.gt_threshold,
        max_n_gt=args.max_n_gt,
    )
    print("TIMELINE_VISUALIZATION_COMPLETE")
