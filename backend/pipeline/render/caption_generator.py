"""
ClipSense AI - Karaoke Caption Generator

Generates high-engagement, word-by-word highlighted karaoke subtitles (ASS format)
and burns them directly into video clips using FFmpeg.
Based on the production subtitle engine from backend/main.py.
"""

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pysubs2

logger = logging.getLogger(__name__)


def generate_karaoke_ass(
    transcript_words: List[Any],
    clip_start: float,
    clip_end: float,
    output_ass_path: Union[str, Path],
    max_words: int = 4,
    font_name: str = "Impact",
    font_size: Optional[int] = None,
    play_res_x: int = 1080,
    play_res_y: int = 1920,
    highlight_color_bgr: tuple = (0, 215, 255),  # Yellow/Gold: B=0, G=215, R=255
    text_color_bgr: tuple = (255, 255, 255),      # White
) -> bool:
    """
    Builds an ASS subtitle file with karaoke color transitions (\\k<centiseconds>)
    for words within [clip_start, clip_end].
    Accepts list of dicts or TimestampedWord instances.
    """
    output_ass = Path(output_ass_path).resolve()
    output_ass.parent.mkdir(parents=True, exist_ok=True)

    # Filter words overlapping the clip interval
    clip_segments = []
    for w in transcript_words:
        if isinstance(w, dict):
            word = str(w.get("word", "")).strip()
            s = w.get("start")
            e = w.get("end")
        else:
            word = str(getattr(w, "word", "")).strip()
            s = getattr(w, "start", None)
            e = getattr(w, "end", None)

        if not word or s is None or e is None:
            continue
        try:
            s_flt = float(s)
            e_flt = float(e)
            if e_flt > clip_start and s_flt < clip_end:
                clip_segments.append({"word": word, "start": s_flt, "end": e_flt})
        except (ValueError, TypeError):
            continue

    if not clip_segments:
        logger.warning("No transcript words found between %.2f and %.2f for subtitles.", clip_start, clip_end)
        return False

    # Group words into short lines for rapid readability
    subtitle_groups = []
    current_words = []
    current_start = None

    for seg in clip_segments:
        word = seg["word"]
        seg_start = seg["start"]
        seg_end = seg["end"]

        start_rel = max(0.0, seg_start - clip_start)
        end_rel = max(0.0, seg_end - clip_start)

        if end_rel <= start_rel:
            end_rel = start_rel + 0.25

        if not current_words:
            current_start = start_rel
            current_words = [(word, start_rel, end_rel)]
        elif len(current_words) >= max_words:
            subtitle_groups.append((current_start, current_words))
            current_words = [(word, start_rel, end_rel)]
            current_start = start_rel
        else:
            current_words.append((word, start_rel, end_rel))

    if current_words:
        subtitle_groups.append((current_start, current_words))

    if not subtitle_groups:
        return False

    subs = pysubs2.SSAFile()
    subs.info["WrapStyle"] = 0
    subs.info["ScaledBorderAndShadow"] = "yes"
    subs.info["PlayResX"] = play_res_x
    subs.info["PlayResY"] = play_res_y
    subs.info["ScriptType"] = "v4.00+"

    effective_fontsize = font_size if font_size is not None else (95 if play_res_y >= 1920 else 55)
    effective_margin_v = 300 if play_res_y >= 1920 else int(play_res_y * 0.12)

    style_name = "ClipSenseKaraoke"
    new_style = pysubs2.SSAStyle()
    # Font styling: Bold display font (Impact is pre-installed across Windows)
    new_style.fontname = font_name
    new_style.fontsize = effective_fontsize
    new_style.bold = True
    # In pysubs2 Color: r, g, b, a
    new_style.secondarycolor = pysubs2.Color(text_color_bgr[2], text_color_bgr[1], text_color_bgr[0])
    new_style.primarycolor = pysubs2.Color(highlight_color_bgr[2], highlight_color_bgr[1], highlight_color_bgr[0])
    new_style.outline = 3.0
    new_style.shadow = 2.5
    new_style.outlinecolor = pysubs2.Color(0, 0, 0, 0)
    new_style.shadowcolor = pysubs2.Color(0, 0, 0, 160)
    new_style.alignment = pysubs2.Alignment.BOTTOM_CENTER  # Bottom-center
    new_style.marginl = 50
    new_style.marginr = 50
    new_style.marginv = effective_margin_v
    new_style.spacing = 1.0

    subs.styles[style_name] = new_style
    subs.styles["Default"] = new_style  # Fallback insurance

    for group_start, words_with_timing in subtitle_groups:
        group_end = words_with_timing[-1][2]
        text_parts = []
        for word, word_start, word_end in words_with_timing:
            duration_cs = max(1, int(round((word_end - word_start) * 100)))
            text_parts.append(f"{{\\k{duration_cs}}}{word.upper()}")

        subtitle_text = " ".join(text_parts)
        start_time = pysubs2.make_time(s=group_start)
        end_time = pysubs2.make_time(s=group_end)
        line = pysubs2.SSAEvent(
            start=start_time, end=end_time, text=subtitle_text, style=style_name
        )
        subs.events.append(line)

    subs.save(str(output_ass))
    return True


def burn_subtitles_ffmpeg(
    video_path: Union[str, Path],
    ass_path: Union[str, Path],
    output_path: Union[str, Path],
    ffmpeg_bin: str = "ffmpeg",
    timeout_sec: float = 120.0,
) -> bool:
    """
    Burns an ASS subtitle file into a video using FFmpeg's ass filter.
    Handles Windows path separators and spaces safely.
    """
    v_in = Path(video_path).resolve()
    v_out = Path(output_path).resolve()
    ass_file = Path(ass_path).resolve()

    if not v_in.exists() or not ass_file.exists():
        logger.error("Missing input video (%s) or subtitle file (%s)", v_in, ass_file)
        return False

    v_out.parent.mkdir(parents=True, exist_ok=True)

    # Use relative filename inside the directory to bypass Windows colon escaping in FFmpeg filter
    work_dir = ass_file.parent
    rel_ass_name = ass_file.name

    cmd = [
        ffmpeg_bin,
        "-y",
        "-nostdin",
        "-i", str(v_in),
        "-vf", f"ass={rel_ass_name}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        "-c:a", "copy",
        str(v_out),
    ]

    try:
        res = subprocess.run(
            cmd,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if res.returncode == 0 and v_out.exists() and v_out.stat().st_size > 0:
            return True
        else:
            logger.warning("FFmpeg ass filter failed with code %d: %s", res.returncode, res.stderr[-300:] if res.stderr else "")
            return False
    except Exception as ex:
        logger.exception("Error executing FFmpeg subtitle burning: %s", ex)
        return False
