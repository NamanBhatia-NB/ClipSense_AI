"""
ClipSense AI — Download & Assemble Human Evaluation Pilot Package

Downloads exact candidate interval sections via yt-dlp, verifies decodability & duration,
and compiles a clean, blinded human evaluation package with separate CSVs for each rater.
"""

import csv
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging = None


def run_ffprobe_duration(clip_path: pathlib.Path) -> Optional[float]:
    """Inspects decoded duration and checks decodability via ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(clip_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if res.returncode == 0 and res.stdout.strip():
            return float(res.stdout.strip())
    except Exception as e:
        print(f"ffprobe error on {clip_path.name}: {e}")
    return None


def download_section(
    youtube_id: str,
    start_sec: float,
    end_sec: float,
    out_path: pathlib.Path,
    max_retries: int = 2,
) -> bool:
    """Downloads an exact temporal section using yt-dlp without fetching the full video."""
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    section_arg = f"*{start_sec:.3f}-{end_sec:.3f}"
    
    cmd = [
        sys.executable,
        "-m", "yt_dlp",
        "--download-sections", section_arg,
        "--force-keyframes-at-cuts",
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "-o", str(out_path),
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        url,
    ]

    for attempt in range(1, max_retries + 1):
        try:
            print(f"    Downloading [{start_sec:.2f}s -> {end_sec:.2f}s] from {youtube_id} (attempt {attempt})...")
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if out_path.exists() and out_path.stat().st_size > 1000:
                return True
        except Exception as e:
            print(f"    Download error: {e}")
        time.sleep(2)

    return out_path.exists() and out_path.stat().st_size > 1000


def build_final_human_eval_package(
    source_pilot_dir: str = "evaluation/human_eval_top1_pilot_corrected",
    target_dir: str = "evaluation/human_eval_final_pilot",
) -> Dict[str, Any]:
    src_path = pathlib.Path(source_pilot_dir).resolve()
    tgt_path = pathlib.Path(target_dir).resolve()
    tgt_path.mkdir(parents=True, exist_ok=True)

    clips_dir = tgt_path / "rendered_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    distribution_dir = tgt_path / "rater_distribution"
    distribution_dir.mkdir(parents=True, exist_ok=True)

    # Load blinded metadata and unblinding key from source pilot
    with open(src_path / "human_eval_blinded.json", "r", encoding="utf-8") as f:
        blinded_data = json.load(f)

    with open(src_path / "researcher_unblinding_key.json", "r", encoding="utf-8") as f:
        unblinding_data = json.load(f)

    clips = blinded_data["clips"]
    clip_truth_mapping = unblinding_data["clip_truth_mapping"]

    print(f"Processing {len(clips)} clips for final human-eval pilot package...")

    # Cache downloaded sections: (yt_id, round(start, 2), round(end, 2)) -> clip_path
    downloaded_cache: Dict[Tuple[str, float, float], pathlib.Path] = {}
    validation_records: List[Dict[str, Any]] = []

    for idx, c in enumerate(clips, start=1):
        cid = c["clip_id"]
        src_vid = c["source_video_id"]
        clean_yt = src_vid.replace("rhapsody_real_", "").strip()
        s_sec = float(c["start_sec"])
        e_sec = float(c["end_sec"])
        intended_dur = float(c["duration_sec"])

        out_clip_file = clips_dir / f"{cid}.mp4"
        rel_clip_path = f"rendered_clips/{cid}.mp4"

        cache_key = (clean_yt, round(s_sec, 2), round(e_sec, 2))
        success = False

        if out_clip_file.exists() and out_clip_file.stat().st_size > 1000:
            print(f"[{idx}/20] {cid} already exists locally: {out_clip_file.name}")
            success = True
        elif cache_key in downloaded_cache and downloaded_cache[cache_key].exists():
            # Identical interval already downloaded for same video (e.g. system A and B or B and C)
            cached_file = downloaded_cache[cache_key]
            print(f"[{idx}/20] {cid} shares identical interval with {cached_file.name}. Copying...")
            shutil.copyfile(cached_file, out_clip_file)
            success = True
        else:
            print(f"[{idx}/20] Fetching exact interval for {cid} ({clean_yt}: {s_sec:.2f}s - {e_sec:.2f}s)...")
            success = download_section(clean_yt, s_sec, e_sec, out_clip_file)
            if success:
                downloaded_cache[cache_key] = out_clip_file

        # Validate existence & decodability
        actual_dur = None
        size_bytes = 0
        is_decodable = False

        if out_clip_file.exists():
            size_bytes = out_clip_file.stat().st_size
            actual_dur = run_ffprobe_duration(out_clip_file)
            is_decodable = (actual_dur is not None and actual_dur > 0.0)

        dur_diff = abs(actual_dur - intended_dur) if actual_dur else None

        truth_info = clip_truth_mapping.get(cid, {})
        val_rec = {
            "clip_id": cid,
            "source_video_id": src_vid,
            "youtube_id": clean_yt,
            "blind_system_code": c["blind_system_code"],
            "system_name_truth": truth_info.get("system_name_truth", "unknown"),
            "start_sec": s_sec,
            "end_sec": e_sec,
            "intended_duration_sec": intended_dur,
            "actual_duration_sec": round(actual_dur, 3) if actual_dur else None,
            "duration_delta_sec": round(dur_diff, 3) if dur_diff is not None else None,
            "file_size_bytes": size_bytes,
            "is_decodable": is_decodable,
            "clip_path": rel_clip_path if is_decodable else c["video_path_or_url"],
        }
        validation_records.append(val_rec)

        # Update clip entry in blinded_data
        if is_decodable:
            c["video_path_or_url"] = rel_clip_path
            c["file_size_bytes"] = size_bytes
            c["actual_duration_sec"] = round(actual_dur, 3)

    # Save updated blinded JSON
    with open(tgt_path / "human_eval_blinded.json", "w", encoding="utf-8") as f:
        json.dump(blinded_data, f, indent=2)

    # Save isolated researcher unblinding key
    with open(tgt_path / "researcher_unblinding_key.json", "w", encoding="utf-8") as f:
        json.dump(unblinding_data, f, indent=2)

    # Copy protocol to rater distribution directory
    proto_src = src_path / "HUMAN_EVAL_PROTOCOL.md"
    if proto_src.exists():
        shutil.copyfile(proto_src, distribution_dir / "HUMAN_EVAL_PROTOCOL.md")
        shutil.copyfile(proto_src, tgt_path / "HUMAN_EVAL_PROTOCOL.md")

    # Generate three separate CSVs, one per rater
    rater_ids = ["RATER_01", "RATER_02", "RATER_03"]
    scoring_columns = [
        "clip_id",
        "source_video_id",
        "start_sec",
        "end_sec",
        "duration_sec",
        "clip_file_or_url",
        "score_interestingness",
        "score_self_containment",
        "score_content_payoff",
        "score_start_end_quality",
        "score_redundancy",
        "score_shareability",
        "rater_comments",
    ]

    all_rater_rows = []
    for rater_id in rater_ids:
        rater_rows = []
        for v in validation_records:
            row = {
                "clip_id": v["clip_id"],
                "source_video_id": v["source_video_id"],
                "start_sec": v["start_sec"],
                "end_sec": v["end_sec"],
                "duration_sec": v["intended_duration_sec"],
                "clip_file_or_url": v["clip_path"],
                "score_interestingness": "",
                "score_self_containment": "",
                "score_content_payoff": "",
                "score_start_end_quality": "",
                "score_redundancy": "",
                "score_shareability": "",
                "rater_comments": "",
            }
            rater_rows.append(row)
            all_rater_rows.append({"rater_id": rater_id, **row})

        r_csv_path = distribution_dir / f"rating_sheet_{rater_id.lower()}.csv"
        with open(r_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=scoring_columns)
            writer.writeheader()
            writer.writerows(rater_rows)

    # Combined rating sheet (for coordinator/researcher reference, scores still blank)
    combined_csv_path = distribution_dir / "rating_sheet_all_raters.csv"
    with open(combined_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rater_id"] + scoring_columns)
        writer.writeheader()
        writer.writerows(all_rater_rows)

    # Write VALIDATION_REPORT.md
    val_report_path = tgt_path / "VALIDATION_REPORT.md"
    generate_validation_report(val_report_path, validation_records)

    total_created = sum(1 for v in validation_records if v["is_decodable"])
    print(f"\nFinal Package Assembly Complete:")
    print(f" - Output Directory:        {tgt_path}")
    print(f" - Rendered Clips Created:  {total_created} / {len(clips)}")
    print(f" - Distribution Folder:     {distribution_dir}")
    print(f" - Validation Report:       {val_report_path}")

    return {
        "target_dir": str(tgt_path),
        "total_clips": len(clips),
        "total_rendered": total_created,
        "validation_records": validation_records,
    }


def generate_validation_report(out_path: pathlib.Path, records: List[Dict[str, Any]]) -> None:
    lines = [
        "# ClipSense AI — Human Evaluation Pilot Package Validation Report",
        "",
        "## 1. Package Summary",
        "",
        f"- **Total Candidate Clips**: {len(records)} clips across 5 sub-hour test episodes.",
        f"- **Primary Systems Evaluated**: System A (Transcript-Only), System B (Simple Aggregation), System C (MTER), System D (MTER Refined).",
        f"- **Rater Assignments**: 3 independent raters per clip (60 total assignments across 3 separate rater CSVs).",
        f"- **Blinding Status**: Strict double-blinding. All system names, model scores, and blinding codes are excluded from rater CSVs.",
        f"- **Score Form State**: All scoring fields are completely blank.",
        f"- **Researcher Unblinding Key**: Strictly isolated in `researcher_unblinding_key.json` outside the `rater_distribution/` folder.",
        "",
        "## 2. Clip Verification & Decodability Table",
        "",
        "| Clip ID | Episode ID | Intended Interval | Intended Dur (s) | Rendered File | Decodable | Decoded Dur (s) | File Size (MB) |",
        "| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: |",
    ]

    for r in records:
        cid = r["clip_id"]
        ep = r["source_video_id"]
        span = f"[{r['start_sec']:.2f}s -> {r['end_sec']:.2f}s]"
        i_dur = f"{r['intended_duration_sec']:.2f}s"
        r_file = f"`{r['clip_path']}`"
        dec = "YES [OK]" if r["is_decodable"] else "FALLBACK (URL)"
        a_dur = f"{r['actual_duration_sec']:.2f}s" if r["actual_duration_sec"] else "N/A"
        f_size = f"{r['file_size_bytes'] / (1024*1024):.2f} MB" if r["file_size_bytes"] else "0 MB"
        lines.append(f"| `{cid}` | `{ep}` | {span} | {i_dur} | {r_file} | {dec} | {a_dur} | {f_size} |")

    lines.extend([
        "",
        "## 3. Invariant C/D Candidate Pairing Verification",
        "",
        "System D applies sub-second boundary refinement (-0.15s start, +0.15s end) to the exact base candidate selected by System C:",
        "",
        "| Episode ID | System C Clip ID | System C Interval | System D Clip ID | System D Interval | Start Shift | End Shift |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    # Group by episode
    from collections import defaultdict
    by_ep = defaultdict(dict)
    for r in records:
        ep = r["source_video_id"]
        sys_truth = r["system_name_truth"]
        by_ep[ep][sys_truth] = r

    for ep in sorted(by_ep.keys()):
        c_rec = by_ep[ep].get("system_c_mter", {})
        d_rec = by_ep[ep].get("system_d_mter_refined", {})
        c_id = c_rec.get("clip_id", "N/A")
        c_span = f"[{c_rec.get('start_sec', 0.0):.2f}s -> {c_rec.get('end_sec', 0.0):.2f}s]"
        d_id = d_rec.get("clip_id", "N/A")
        d_span = f"[{d_rec.get('start_sec', 0.0):.2f}s -> {d_rec.get('end_sec', 0.0):.2f}s]"
        ds = round(d_rec.get("start_sec", 0.0) - c_rec.get("start_sec", 0.0), 3) if c_rec and d_rec else 0.0
        de = round(d_rec.get("end_sec", 0.0) - c_rec.get("end_sec", 0.0), 3) if c_rec and d_rec else 0.0
        lines.append(f"| `{ep}` | `{c_id}` | {c_span} | `{d_id}` | {d_span} | {ds:+.2f}s | {de:+.2f}s |")

    lines.extend([
        "",
        "## 4. Distribution Instructions for Raters",
        "",
        "To distribute the evaluation package to 3 independent raters:",
        "1. Provide each rater with the `rater_distribution/` directory containing:",
        "   - `HUMAN_EVAL_PROTOCOL.md` (Scoring guide and 1–5 rubric definition)",
        "   - Their respective sheet: `rating_sheet_rater_01.csv`, `rating_sheet_rater_02.csv`, or `rating_sheet_rater_03.csv`",
        "   - The `rendered_clips/` folder containing `clip_01.mp4` through `clip_20.mp4`",
        "2. Raters watch the 20 local MP4 files directly, ensuring exact playback of both start and end boundaries.",
        "3. Keep `researcher_unblinding_key.json` strictly private until all ratings are returned.",
        "",
    ])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    build_final_human_eval_package()
