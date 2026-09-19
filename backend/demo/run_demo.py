"""
ClipSense Demonstration and Observability Runner (W6/W1-W6)

Demonstrates the step-by-step multimodal extraction, independent expert evidence
generation, MTER consensus reasoning, and optional evaluation comparison against
reference annotations.

Usage:
    python -m demo.run_demo --video <path_to_video> --output-dir <path_to_dir> [options]

Options:
    --reference <path_to_dataset_json>: Optional reference annotation for evaluation.
    --use-cache: Reuses existing cached artifacts if present to avoid re-running WhisperX/LLM.
    --mock-llm: Uses deterministic mock LLM for offline testing without API keys.
    --device: 'cpu' or 'cuda' (default: 'cpu').
"""

import argparse
import html
import json
import logging
import os
import pathlib
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure backend root is on sys.path
backend_dir = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(backend_dir))

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.config import config
from core.llm import LLMClient, MockLLMClient, get_llm_client
from core.schemas import (
    CommonEvidenceBundle,
    ConversationData,
    ExpertEvidenceBundle,
    MTEROutput,
    ProsodyData,
    TranscriptData,
    VisualData,
)
from evaluation.dataset import load_benchmark_dataset
from evaluation.metrics import match_spans_hungarian
from evaluation.schemas import EvaluationSpan, ReferenceHighlight
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

logger = logging.getLogger("clipsense.demo")


class DemoMockLLMClient(MockLLMClient):
    """Deterministic mock client used when --mock-llm is active or API key is absent."""

    def generate_structured(
        self, prompt: str, response_schema: Any, system_instruction: Optional[str] = None
    ) -> Any:
        import re

        if response_schema == TranscriptLLMResponse:
            indices = [int(m) for m in re.findall(r"\[(\d+)\]", prompt)]
            candidates = []
            if len(indices) >= 20:
                start_i = indices[min(2, len(indices) - 1)]
                end_i = indices[min(len(indices) - 3, 50)]
                candidates.append(
                    TranscriptSemanticCandidate(
                        start_word_index=start_i,
                        end_word_index=end_i,
                        evidence_type="explanatory_claim",
                        explanation="Thematic narrative claim extracted from spoken dialogue.",
                        confidence_estimate=0.88,
                        topic_transition=False,
                        contextual_completeness=0.86,
                        semantic_importance=0.84,
                        self_contained=True,
                    )
                )
            return TranscriptLLMResponse(candidates=candidates)

        elif response_schema == ConversationLLMResponse:
            match = re.search(r"\[([\d\.]+)s\s*->\s*([\d\.]+)s\]", prompt)
            if match:
                s_est = float(match.group(1))
                e_est = max(s_est + 15.0, min(s_est + 27.0, float(match.group(2))))
            else:
                s_est = 0.0
                e_est = 25.0
            return ConversationLLMResponse(
                candidates=[
                    ConversationLLMCandidate(
                        start_time_estimate=s_est,
                        end_time_estimate=e_est,
                        evidence_type="monologue_thematic_unit",
                        explanation="Opening monologue discussion block with cohesive topic pacing.",
                        confidence_estimate=0.85,
                        speaker_interaction=False,
                        discourse_unit_complete=True,
                        pause_boundary_supported=True,
                        discourse_phase="setup_development_payoff",
                    )
                ]
            )

        return super().generate_structured(prompt, response_schema, system_instruction)


def generate_html_report(
    video_path: str,
    video_size_mb: float,
    total_duration: float,
    extraction_summary: Dict[str, Any],
    expert_bundles: Dict[str, ExpertEvidenceBundle],
    mter_output: MTEROutput,
    mter_ledger: Any,
    evaluation_info: Optional[Dict[str, Any]],
    output_html_path: pathlib.Path,
):
    """
    Builds a self-contained, human-readable HTML demonstration report.
    """
    selected = mter_output.candidates[0] if mter_output.candidates else None
    run_type = "demonstration"
    if evaluation_info:
        run_type = evaluation_info.get("sample_type", "demonstration")

    # Colors for timeline modalities
    colors = {
        "transcript": "#38bdf8",    # Sky blue
        "conversation": "#4ade80",  # Green
        "visual": "#fbbf24",        # Amber
        "prosody": "#c084fc",       # Purple
        "mter": "#f43f5e",          # Rose / Red
        "reference": "#2dd4bf",     # Teal
    }

    # Build timeline bars HTML
    def make_bar(start: float, end: float, color: str, title: str, label: str) -> str:
        dur = max(0.01, total_duration)
        left = (start / dur) * 100.0
        width = max(0.5, ((end - start) / dur) * 100.0)
        return (
            f'<div class="timeline-bar" style="left: {left:.2f}%; width: {width:.2f}%; background: {color};" '
            f'title="{html.escape(title)}">{html.escape(label)}</div>'
        )

    t_bars = "".join(
        make_bar(
            p.start_time,
            p.end_time,
            colors["transcript"],
            f"Transcript: {p.start_time:.1f}s - {p.end_time:.1f}s (conf: {p.confidence_estimate:.2f}) | {p.explanation}",
            f"{p.start_time:.1f}s",
        )
        for p in expert_bundles["transcript"].proposals
    )

    c_bars = "".join(
        make_bar(
            p.start_time,
            p.end_time,
            colors["conversation"],
            f"Conversation: {p.start_time:.1f}s - {p.end_time:.1f}s (conf: {p.confidence_estimate:.2f}) | {p.explanation}",
            f"{p.start_time:.1f}s",
        )
        for p in expert_bundles["conversation"].proposals
    )

    v_bars = "".join(
        make_bar(
            p.start_time,
            p.end_time,
            colors["visual"],
            f"Visual: {p.start_time:.1f}s - {p.end_time:.1f}s (conf: {p.confidence_estimate:.2f}) | {p.explanation}",
            f"{p.start_time:.1f}s",
        )
        for p in expert_bundles["visual"].proposals
    )

    p_bars = "".join(
        make_bar(
            p.start_time,
            p.end_time,
            colors["prosody"],
            f"Prosody: {p.start_time:.1f}s - {p.end_time:.1f}s (conf: {p.confidence_estimate:.2f}) | {p.explanation}",
            f"{p.start_time:.1f}s",
        )
        for p in expert_bundles["prosody"].proposals
    )

    mter_bars = "".join(
        make_bar(
            c.proposed_start,
            c.proposed_end,
            colors["mter"],
            f"MTER Candidate: {c.proposed_start:.2f}s - {c.proposed_end:.2f}s (score: {c.confidence_estimate:.3f})",
            f"MTER [{c.proposed_start:.1f}s - {c.proposed_end:.1f}s]",
        )
        for c in mter_output.candidates
    )

    ref_bars = ""
    if evaluation_info and evaluation_info.get("reference_spans"):
        ref_bars = "".join(
            make_bar(
                r["start_time"],
                r["end_time"],
                colors["reference"],
                f"Reference: {r['start_time']:.2f}s - {r['end_time']:.2f}s | {r.get('label', '')}",
                f"Ref [{r['start_time']:.1f}s]",
            )
            for r in evaluation_info["reference_spans"]
        )

    # Build expert tables
    def make_table(bundle: ExpertEvidenceBundle) -> str:
        if not bundle.proposals:
            return '<div class="empty-notice">No temporal proposals emitted by this expert.</div>'
        rows = []
        for p in bundle.proposals:
            rows.append(
                f"<tr>"
                f"<td><code>{p.proposal_id}</code></td>"
                f"<td><b>{p.start_time:.2f}s &rarr; {p.end_time:.2f}s</b></td>"
                f"<td>{(p.end_time - p.start_time):.2f}s</td>"
                f"<td><span class=\"badge badge-conf\">{p.confidence_estimate:.2f}</span></td>"
                f"<td><code>{p.evidence_type}</code></td>"
                f"<td>{html.escape(p.explanation)}</td>"
                f"</tr>"
            )
        return (
            '<table class="data-table">'
            "<thead><tr><th>ID</th><th>Interval</th><th>Duration</th><th>Confidence</th><th>Evidence Type</th><th>Explanation</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    # Badge HTML
    if run_type == "research":
        badge_html = '<span class="status-badge badge-research">RESEARCH BENCHMARK EVALUATION</span>'
    elif run_type == "development":
        badge_html = (
            '<span class="status-badge badge-dev">DEVELOPMENT / DEBUG EVALUATION</span>'
            '<p class="badge-subtitle">Development reference interval used for verification only; not research ground truth.</p>'
        )
    else:
        badge_html = (
            '<span class="status-badge badge-demo">PREDICTION-ONLY DEMONSTRATION</span>'
            '<p class="badge-subtitle">No reference annotations provided; metrics omitted.</p>'
        )

    eval_section_html = ""
    if evaluation_info and evaluation_info.get("matched_metrics"):
        m = evaluation_info["matched_metrics"][0]
        eval_section_html = f"""
        <section class="card">
            <h2>10. Evaluation Comparison ({evaluation_info.get('annotation_source', 'Reference')})</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-val">{m['temporal_iou']:.3f}</div>
                    <div class="metric-lbl">Temporal IoU</div>
                </div>
                <div class="metric-card">
                    <div class="metric-val">{m['start_error_sec']:.2f}s</div>
                    <div class="metric-lbl">Start Boundary Error</div>
                </div>
                <div class="metric-card">
                    <div class="metric-val">{m['end_error_sec']:.2f}s</div>
                    <div class="metric-lbl">End Boundary Error</div>
                </div>
                <div class="metric-card">
                    <div class="metric-val">{m['duration_error_sec']:.2f}s</div>
                    <div class="metric-lbl">Duration Error</div>
                </div>
                <div class="metric-card">
                    <div class="metric-val">{'PASS' if m['hit_at_2s'] else 'FAIL'}</div>
                    <div class="metric-lbl">Hit @ 2.0s Tolerance</div>
                </div>
            </div>
            <div class="callout callout-info">
                <strong>Protocol:</strong> Evaluated using Hungarian bipartite matching ($\tau_{{\\text{{match}}}} = 0.10$). 
                Matched predicted span <code>{m['prediction_span_id']}</code> [{m['pred_start']:.2f}s &rarr; {m['pred_end']:.2f}s]
                against reference span <code>{m['reference_span_id']}</code> [{m['ref_start']:.2f}s &rarr; {m['ref_end']:.2f}s].
            </div>
        </section>
        """
    else:
        eval_section_html = """
        <section class="card">
            <h2>10. Evaluation</h2>
            <div class="callout callout-neutral">
                <strong>Prediction-only demonstration:</strong> No ground-truth annotations were supplied for this run. Evaluation metrics are not computed or invented.
            </div>
        </section>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ClipSense AI — Pipeline Demonstration Report</title>
    <style>
        :root {{
            --bg: #0f172a;
            --card-bg: #1e293b;
            --border: #334155;
            --accent: #38bdf8;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --green: #4ade80;
            --amber: #fbbf24;
            --purple: #c084fc;
            --rose: #f43f5e;
            --teal: #2dd4bf;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            padding: 30px 20px;
            line-height: 1.5;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
        }}
        h1 {{ font-size: 1.75rem; font-weight: 700; color: #fff; }}
        .header-meta {{ color: var(--text-muted); font-size: 0.9rem; margin-top: 4px; }}
        .status-badge {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }}
        .badge-demo {{ background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid #38bdf8; }}
        .badge-dev {{ background: rgba(251, 191, 36, 0.15); color: #fbbf24; border: 1px solid #fbbf24; }}
        .badge-research {{ background: rgba(74, 222, 128, 0.15); color: #4ade80; border: 1px solid #4ade80; }}
        .badge-subtitle {{ font-size: 0.75rem; color: var(--text-muted); margin-top: 4px; text-align: right; }}
        
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }}
        h2 {{ font-size: 1.25rem; font-weight: 600; margin-bottom: 16px; color: var(--accent); }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 16px;
        }}
        .stat-box {{
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
        }}
        .stat-val {{ font-size: 1.8rem; font-weight: 700; color: #fff; }}
        .stat-lbl {{ color: var(--text-muted); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }}
        
        /* Timeline Visualizer */
        .timeline-container {{
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px 16px;
            margin: 16px 0;
        }}
        .timeline-track-lbl {{
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            display: flex;
            justify-content: space-between;
        }}
        .timeline-track {{
            position: relative;
            height: 28px;
            background: rgba(51, 65, 85, 0.3);
            border-radius: 6px;
            margin-bottom: 14px;
            overflow: hidden;
        }}
        .timeline-bar {{
            position: absolute;
            top: 2px;
            bottom: 2px;
            border-radius: 4px;
            display: flex;
            align-items: center;
            padding: 0 6px;
            font-size: 0.75rem;
            font-weight: 600;
            color: #0f172a;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            cursor: pointer;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }}
        .timeline-ruler {{
            display: flex;
            justify-content: space-between;
            color: var(--text-muted);
            font-size: 0.75rem;
            border-top: 1px dashed var(--border);
            padding-top: 6px;
        }}
        
        /* Candidate Card */
        .candidate-card {{
            background: rgba(244, 63, 94, 0.05);
            border: 1px solid var(--rose);
            border-radius: 8px;
            padding: 20px;
            margin-top: 16px;
        }}
        .candidate-title {{ font-size: 1.1rem; font-weight: 700; color: #fff; margin-bottom: 8px; }}
        .support-bars {{ margin: 12px 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }}
        .support-item {{ font-size: 0.85rem; }}
        .progress-bar {{ background: rgba(255,255,255,0.1); height: 8px; border-radius: 4px; overflow: hidden; margin-top: 4px; }}
        .progress-fill {{ height: 100%; border-radius: 4px; }}
        
        /* Tables */
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
            margin-top: 10px;
        }}
        .data-table th, .data-table td {{
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        .data-table th {{ background: rgba(51, 65, 85, 0.4); color: var(--text-muted); font-weight: 600; }}
        .data-table tr:hover {{ background: rgba(255,255,255,0.02); }}
        .badge-conf {{
            background: rgba(56, 189, 248, 0.15);
            color: #38bdf8;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 700;
        }}
        .callout {{
            padding: 14px 18px;
            border-radius: 8px;
            font-size: 0.9rem;
            margin-top: 12px;
        }}
        .callout-info {{ background: rgba(56, 189, 248, 0.1); border-left: 4px solid var(--accent); }}
        .callout-neutral {{ background: rgba(148, 163, 184, 0.1); border-left: 4px solid var(--text-muted); }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin: 16px 0;
        }}
        .metric-card {{
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 14px;
            text-align: center;
        }}
        .metric-val {{ font-size: 1.6rem; font-weight: 700; color: #fff; }}
        .metric-lbl {{ font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; margin-top: 4px; }}
        .empty-notice {{ color: var(--text-muted); font-style: italic; padding: 12px 0; }}
        code {{ background: rgba(0,0,0,0.3); padding: 2px 4px; border-radius: 4px; font-family: monospace; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>ClipSense AI &mdash; End-to-End Multimodal Demonstration</h1>
                <div class="header-meta">
                    Input: <code>{pathlib.Path(video_path).name}</code> &bull; Size: {video_size_mb:.2f} MB &bull; Duration: {total_duration:.2f}s
                </div>
            </div>
            <div>
                {badge_html}
            </div>
        </header>

        <!-- Stage 2: Extraction -->
        <section class="card">
            <h2>2. Multimodal Extraction Summary (W2)</h2>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-val">{extraction_summary.get('word_count', 0)}</div>
                    <div class="stat-lbl">Spoken Words (WhisperX)</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val">{extraction_summary.get('sampled_frames', 0)}</div>
                    <div class="stat-lbl">Sampled Frames ({extraction_summary.get('scene_boundaries', 0)} Scene Cuts)</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val">{extraction_summary.get('prosody_windows', 0)}</div>
                    <div class="stat-lbl">Acoustic Windows (Librosa F0/RMS)</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val">{extraction_summary.get('conversation_turns', 0)}</div>
                    <div class="stat-lbl">Conversation Turns / Segments</div>
                </div>
            </div>
        </section>

        <!-- Visual Timeline -->
        <section class="card">
            <h2>4-Modality Evidence & Consensus Timeline</h2>
            <div class="timeline-container">
                <div class="timeline-track-lbl">
                    <span>1. Spoken Transcript Evidence</span>
                    <span style="color: {colors['transcript']}; font-weight: 600;">{len(expert_bundles['transcript'].proposals)} Proposals</span>
                </div>
                <div class="timeline-track">{t_bars or '<span class="empty-notice" style="padding-left:10px;">None</span>'}</div>

                <div class="timeline-track-lbl">
                    <span>2. Conversation Dynamics Evidence</span>
                    <span style="color: {colors['conversation']}; font-weight: 600;">{len(expert_bundles['conversation'].proposals)} Proposals</span>
                </div>
                <div class="timeline-track">{c_bars or '<span class="empty-notice" style="padding-left:10px;">None</span>'}</div>

                <div class="timeline-track-lbl">
                    <span>3. Visual Shifts & Activity Evidence</span>
                    <span style="color: {colors['visual']}; font-weight: 600;">{len(expert_bundles['visual'].proposals)} Proposals</span>
                </div>
                <div class="timeline-track">{v_bars or '<span class="empty-notice" style="padding-left:10px;">None</span>'}</div>

                <div class="timeline-track-lbl">
                    <span>4. Acoustic Prosody & Loudness Peaks</span>
                    <span style="color: {colors['prosody']}; font-weight: 600;">{len(expert_bundles['prosody'].proposals)} Proposals</span>
                </div>
                <div class="timeline-track">{p_bars or '<span class="empty-notice" style="padding-left:10px;">None</span>'}</div>

                <div class="timeline-track-lbl">
                    <span>5. Multimodal Reasoner (MTER) Selected Highlights</span>
                    <span style="color: {colors['mter']}; font-weight: 600;">{len(mter_output.candidates)} Candidate(s)</span>
                </div>
                <div class="timeline-track">{mter_bars or '<span class="empty-notice" style="padding-left:10px;">None</span>'}</div>

                {f'<div class="timeline-track-lbl"><span>Reference Annotation</span><span style="color: {colors["reference"]}; font-weight:600;">Ground Truth</span></div><div class="timeline-track">{ref_bars}</div>' if ref_bars else ''}

                <div class="timeline-ruler">
                    <span>0.0s</span>
                    <span>{total_duration * 0.25:.1f}s</span>
                    <span>{total_duration * 0.50:.1f}s</span>
                    <span>{total_duration * 0.75:.1f}s</span>
                    <span>{total_duration:.1f}s</span>
                </div>
            </div>
        </section>

        <!-- Selected Candidate -->
        <section class="card">
            <h2>8 & 9. MTER Selected Candidate Highlights</h2>
            {f'''
            <div class="candidate-card">
                <div class="candidate-title">
                    Top Highlight Candidate: [{selected.proposed_start:.2f}s &rarr; {selected.proposed_end:.2f}s] 
                    <span style="color: var(--text-muted); font-size: 0.9rem; font-weight: 400;">(Duration: {selected.duration:.2f}s)</span>
                </div>
                <p><strong>Decision Summary:</strong> {html.escape(selected.decision_summary)}</p>
                
                <div style="margin-top: 14px;">
                    <strong>Composite Evidence Strength:</strong> 
                    <span class="status-badge badge-demo">{selected.confidence_estimate:.3f} / 1.000</span>
                    &bull; <strong>Agreement IoU:</strong> {selected.temporal_agreement_iou:.3f}
                </div>

                <div class="support-bars">
                    <div class="support-item">
                        <div>Transcript Support: {selected.support_metrics.transcript_support * 100:.1f}%</div>
                        <div class="progress-bar"><div class="progress-fill" style="width: {selected.support_metrics.transcript_support * 100:.1f}%; background: {colors['transcript']};"></div></div>
                    </div>
                    <div class="support-item">
                        <div>Conversation Support: {selected.support_metrics.conversation_support * 100:.1f}%</div>
                        <div class="progress-bar"><div class="progress-fill" style="width: {selected.support_metrics.conversation_support * 100:.1f}%; background: {colors['conversation']};"></div></div>
                    </div>
                    <div class="support-item">
                        <div>Visual Support: {selected.support_metrics.visual_support * 100:.1f}%</div>
                        <div class="progress-bar"><div class="progress-fill" style="width: {selected.support_metrics.visual_support * 100:.1f}%; background: {colors['visual']};"></div></div>
                    </div>
                    <div class="support-item">
                        <div>Prosody Support: {selected.support_metrics.prosody_support * 100:.1f}%</div>
                        <div class="progress-bar"><div class="progress-fill" style="width: {selected.support_metrics.prosody_support * 100:.1f}%; background: {colors['prosody']};"></div></div>
                    </div>
                </div>
            </div>
            ''' if selected else '<div class="empty-notice">No candidate satisfied temporal bounds.</div>'}
        </section>

        <!-- Evaluation Section -->
        {eval_section_html}

        <!-- Expert Detail Tables -->
        <section class="card">
            <h2>3. Transcript Expert Evidence Proposals (W3)</h2>
            {make_table(expert_bundles['transcript'])}
        </section>

        <section class="card">
            <h2>4. Conversation Expert Evidence Proposals (W3)</h2>
            {make_table(expert_bundles['conversation'])}
        </section>

        <section class="card">
            <h2>5. Visual Expert Evidence Proposals (W4)</h2>
            {make_table(expert_bundles['visual'])}
        </section>

        <section class="card">
            <h2>6. Prosody Expert Evidence Proposals (W4)</h2>
            {make_table(expert_bundles['prosody'])}
        </section>
    </div>
</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def run_demonstration(
    video_path: str,
    output_dir: str,
    reference_path: Optional[str] = None,
    use_cache: bool = True,
    device: str = "cpu",
    whisper_model: str = "tiny.en",
    mock_llm: bool = False,
    transcript_window_sec: Optional[float] = None,
):
    """
    Coordinates the 10-stage visible demonstration run.
    """
    t_start = time.time()
    v_path = pathlib.Path(video_path).resolve()
    if not v_path.exists():
        print(f"Error: Input video not found: {video_path}")
        sys.exit(1)

    out_p = pathlib.Path(output_dir).resolve()
    out_p.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("CLIPSENSE AI — END-TO-END PIPELINE DEMO")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. INPUT VIDEO
    # -------------------------------------------------------------------------
    t_stage = time.time()
    v_size_mb = v_path.stat().st_size / (1024 * 1024)
    print("\n1. Input Video")
    print(f"   - File:     {v_path.name}")
    print(f"   - Location: {v_path}")
    print(f"   - Size:     {v_size_mb:.2f} MB")
    print(f"   [✓] Input verified (elapsed: {time.time() - t_stage:.2f}s)")

    # -------------------------------------------------------------------------
    # 2. MULTIMODAL EXTRACTION
    # -------------------------------------------------------------------------
    t_stage = time.time()
    print("\n2. Multimodal Extraction")

    # Fast cache check: check if extraction artifacts exist in out_p or in runs/representative_90s_validation
    # Fast cache check: check if extraction artifacts exist in out_p and actually match this video
    t_file = out_p / "transcript.json"
    c_file = out_p / "conversation.json"
    v_file = out_p / "frames.json"
    p_file = out_p / "prosody.json"
    m_file = out_p / "manifest.json"

    out_p_matches_video = False
    if t_file.exists() and c_file.exists() and v_file.exists() and p_file.exists() and m_file.exists():
        try:
            with open(m_file, "r", encoding="utf-8") as f:
                m_data = json.load(f)
            cached_vid = m_data.get("video_filename", "")
            cached_src = m_data.get("source_video", "")
            if cached_vid == v_path.name or pathlib.Path(cached_src).name == v_path.name:
                out_p_matches_video = True
        except Exception:
            out_p_matches_video = False

    # Only fall back to representative_90s_validation if the input video is the 90s sample
    is_90s_sample = "sample_conversational_90s" in v_path.name.lower()
    if use_cache and not out_p_matches_video and is_90s_sample:
        candidate_cache = backend_dir / "runs" / "representative_90s_validation"
        if candidate_cache.exists() and (candidate_cache / "transcript.json").exists():
            print("   -> Found existing cached extraction artifacts for 90s sample in runs/representative_90s_validation.")
            for fname in [
                "transcript.json",
                "conversation.json",
                "frames.json",
                "prosody.json",
                "manifest.json",
                "transcript_evidence.json",
                "conversation_evidence.json",
                "visual_evidence.json",
                "prosody_evidence.json",
            ]:
                src = candidate_cache / fname
                if src.exists():
                    shutil.copy(src, out_p / fname)
            out_p_matches_video = True

    needs_extract = not (use_cache and out_p_matches_video)

    if needs_extract:
        print("   -> Executing real model inference pipelines...")
        transcript_extractor = TranscriptExtractor(
            model_name=whisper_model,
            device=device,
            compute_type="int8" if device == "cpu" else "float16",
            strict_mode=True,
        )
        extraction_pipeline = MultimodalExtractionPipeline(
            transcript_extractor=transcript_extractor,
        )
        extraction_pipeline.run(
            video_path=str(v_path),
            run_id=out_p.name,
            runs_base_dir=str(out_p.parent),
            force_rerun=True,
        )
    else:
        print("   -> Fast-cache active: Reusing pre-extracted multimodal features.")

    with open(t_file, "r", encoding="utf-8") as f:
        transcript_data = TranscriptData.model_validate_json(f.read())
    with open(c_file, "r", encoding="utf-8") as f:
        conversation_data = ConversationData.model_validate_json(f.read())
    with open(v_file, "r", encoding="utf-8") as f:
        visual_data = VisualData.model_validate_json(f.read())
    with open(p_file, "r", encoding="utf-8") as f:
        prosody_data = ProsodyData.model_validate_json(f.read())

    total_duration = visual_data.duration
    print(f"   - Video Duration:      {total_duration:.2f}s")
    print(f"   - Transcript / WhisperX: {len(transcript_data.words)} words, {len(transcript_data.segments)} segments")
    print(f"   - Video Frames:        {len(visual_data.sampled_frames)} sampled frames, {len(visual_data.scene_boundaries)} cuts")
    print(f"   - Prosody:             {len(prosody_data.windows)} acoustic windows (F0 pitch + RMS energy)")
    print(f"   - Conversation:        {len(conversation_data.turns)} speaker turns, {conversation_data.total_speech_time:.1f}s speech")
    print(f"   [✓] Multimodal Extraction complete (elapsed: {time.time() - t_stage:.2f}s)")

    extraction_summary = {
        "word_count": len(transcript_data.words),
        "sampled_frames": len(visual_data.sampled_frames),
        "scene_boundaries": len(visual_data.scene_boundaries),
        "prosody_windows": len(prosody_data.windows),
        "conversation_turns": len(conversation_data.turns),
    }

    # Setup LLM client
    has_api_key = bool(os.getenv("GEMINI_API_KEY") or config.gemini_api_key)
    if mock_llm or not has_api_key:
        llm_client = DemoMockLLMClient()
    else:
        llm_client = get_llm_client()

    # -------------------------------------------------------------------------
    # 3. TRANSCRIPT EXPERT (W3)
    # -------------------------------------------------------------------------
    can_use_expert_cache = use_cache and out_p_matches_video and not needs_extract

    t_stage = time.time()
    # Compute adaptive window duration to keep total LLM API calls <= 6 for any video duration
    if transcript_window_sec is not None:
        eff_win_sec = float(transcript_window_sec)
        eff_overlap_sec = min(60.0, max(10.0, eff_win_sec * 0.1))
    elif total_duration > 300.0:
        eff_win_sec = max(180.0, total_duration / 5.5)
        eff_overlap_sec = min(60.0, max(15.0, eff_win_sec * 0.1))
    else:
        eff_win_sec = config.transcript.window_duration_sec
        eff_overlap_sec = config.transcript.window_overlap_sec

    print(f"\n3. Transcript Expert (Window: {eff_win_sec:.0f}s, Overlap: {eff_overlap_sec:.0f}s)")
    t_ev_file = out_p / "transcript_evidence.json"
    if can_use_expert_cache and t_ev_file.exists():
        with open(t_ev_file, "r", encoding="utf-8") as f:
            t_evidence = ExpertEvidenceBundle.model_validate_json(f.read())
    else:
        t_expert = TranscriptExpert(
            llm_client=llm_client,
            config=config.transcript,
            window_duration_sec=eff_win_sec,
            window_overlap_sec=eff_overlap_sec,
        )
        try:
            t_evidence = t_expert.evaluate(transcript_data)
        except Exception as e:
            logger.warning(f"Live LLM call failed ({e}). Falling back to DemoMockLLMClient.")
            fallback_client = DemoMockLLMClient()
            t_expert = TranscriptExpert(
                llm_client=fallback_client,
                config=config.transcript,
                window_duration_sec=eff_win_sec,
                window_overlap_sec=eff_overlap_sec,
            )
            t_evidence = t_expert.evaluate(transcript_data)
        with open(t_ev_file, "w", encoding="utf-8") as f:
            f.write(t_evidence.model_dump_json(indent=2))

    print(f"   - Proposal Count: {len(t_evidence.proposals)}")
    for p in t_evidence.proposals:
        print(f"     * [{p.start_time:.2f}s -> {p.end_time:.2f}s] conf={p.confidence_estimate:.2f} ({p.evidence_type}): {p.explanation[:60]}...")
    print(f"   [✓] Transcript Expert complete (elapsed: {time.time() - t_stage:.2f}s)")

    # -------------------------------------------------------------------------
    # 4. CONVERSATION EXPERT (W3)
    # -------------------------------------------------------------------------
    t_stage = time.time()
    print("\n4. Conversation Expert")
    c_ev_file = out_p / "conversation_evidence.json"
    if can_use_expert_cache and c_ev_file.exists():
        with open(c_ev_file, "r", encoding="utf-8") as f:
            c_evidence = ExpertEvidenceBundle.model_validate_json(f.read())
    else:
        c_expert = ConversationExpert(llm_client=llm_client, config=config.conversation)
        try:
            c_evidence = c_expert.evaluate(conversation_data, context={"transcript_data": transcript_data})
        except Exception as e:
            logger.warning(f"Live LLM call failed ({e}). Falling back to DemoMockLLMClient.")
            fallback_client = DemoMockLLMClient()
            c_expert = ConversationExpert(llm_client=fallback_client, config=config.conversation)
            c_evidence = c_expert.evaluate(conversation_data, context={"transcript_data": transcript_data})
        with open(c_ev_file, "w", encoding="utf-8") as f:
            f.write(c_evidence.model_dump_json(indent=2))

    print(f"   - Proposal Count: {len(c_evidence.proposals)}")
    for p in c_evidence.proposals:
        print(f"     * [{p.start_time:.2f}s -> {p.end_time:.2f}s] conf={p.confidence_estimate:.2f} ({p.evidence_type}): {p.explanation[:60]}...")
    print(f"   [✓] Conversation Expert complete (elapsed: {time.time() - t_stage:.2f}s)")

    # -------------------------------------------------------------------------
    # 5. VISUAL EXPERT (W4)
    # -------------------------------------------------------------------------
    t_stage = time.time()
    print("\n5. Visual Expert")
    v_ev_file = out_p / "visual_evidence.json"
    if can_use_expert_cache and v_ev_file.exists():
        with open(v_ev_file, "r", encoding="utf-8") as f:
            v_evidence = ExpertEvidenceBundle.model_validate_json(f.read())
    else:
        v_expert = VisualExpert(config=config.visual)
        v_evidence = v_expert.evaluate(visual_data)
        with open(v_ev_file, "w", encoding="utf-8") as f:
            f.write(v_evidence.model_dump_json(indent=2))

    print(f"   - Proposal Count: {len(v_evidence.proposals)}")
    for p in v_evidence.proposals:
        print(f"     * [{p.start_time:.2f}s -> {p.end_time:.2f}s] conf={p.confidence_estimate:.2f} ({p.evidence_type}): anchor={p.source_metadata.alignment_anchor}")
    print(f"   [✓] Visual Expert complete (elapsed: {time.time() - t_stage:.2f}s)")

    # -------------------------------------------------------------------------
    # 6. PROSODY EXPERT (W4)
    # -------------------------------------------------------------------------
    t_stage = time.time()
    print("\n6. Prosody Expert")
    p_ev_file = out_p / "prosody_evidence.json"
    if can_use_expert_cache and p_ev_file.exists():
        with open(p_ev_file, "r", encoding="utf-8") as f:
            p_evidence = ExpertEvidenceBundle.model_validate_json(f.read())
    else:
        p_expert = ProsodyExpert(config=config.prosody)
        p_evidence = p_expert.evaluate(prosody_data)
        with open(p_ev_file, "w", encoding="utf-8") as f:
            f.write(p_evidence.model_dump_json(indent=2))

    print(f"   - Proposal Count: {len(p_evidence.proposals)}")
    for p in p_evidence.proposals:
        print(f"     * [{p.start_time:.2f}s -> {p.end_time:.2f}s] conf={p.confidence_estimate:.2f} ({p.evidence_type}): {p.explanation[:60]}...")
    print(f"   [✓] Prosody Expert complete (elapsed: {time.time() - t_stage:.2f}s)")

    expert_bundles = {
        "transcript": t_evidence,
        "conversation": c_evidence,
        "visual": v_evidence,
        "prosody": p_evidence,
    }

    # -------------------------------------------------------------------------
    # 7. COMMON EVIDENCE BUNDLE (W5)
    # -------------------------------------------------------------------------
    t_stage = time.time()
    print("\n7. Common Evidence Bundle")
    bundle = build_common_evidence_bundle(
        video_id=out_p.name,
        total_duration=total_duration,
        transcript_evidence=t_evidence,
        conversation_evidence=c_evidence,
        visual_evidence=v_evidence,
        prosody_evidence=p_evidence,
    )
    b_file = out_p / "common_evidence_bundle.json"
    with open(b_file, "w", encoding="utf-8") as f:
        f.write(bundle.model_dump_json(indent=2))

    total_props = sum(len(b.proposals) for b in expert_bundles.values())
    print(f"   - Total Integrated Proposals: {total_props} (4 modalities isolated)")
    print(f"   - Auxiliary Temporal Grid:     {bundle.auxiliary_grid.total_bins} bins (1.0s resolution)")
    print(f"   [✓] Common Evidence Bundle assembled (elapsed: {time.time() - t_stage:.2f}s)")

    # -------------------------------------------------------------------------
    # 8. MULTIMODAL TEMPORAL EVIDENCE REASONER (MTER) (W5)
    # -------------------------------------------------------------------------
    t_stage = time.time()
    print("\n8. MTER (Multimodal Temporal Evidence Reasoner)")
    reasoner = MTERReasoner(config.mter)
    mter_output, mter_ledger = reasoner.reason(bundle)

    out_file = out_p / "mter_output.json"
    ledger_file = out_p / "mter_ledger.json"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(mter_output.model_dump_json(indent=2))
    with open(ledger_file, "w", encoding="utf-8") as f:
        f.write(mter_ledger.model_dump_json(indent=2))

    print(f"   - Event Regions Formed:  {len(mter_ledger.event_regions)}")
    print(f"   - Start Boundary Clusters: {len(mter_ledger.start_clusters)}")
    print(f"   - End Boundary Clusters:   {len(mter_ledger.end_clusters)}")
    print(f"   - Temporal Conflicts:      {len(mter_ledger.conflicts)} detected across regions")
    print(f"   - Evaluated Candidates:    {len(mter_ledger.evaluated_candidates)} intervals")
    print(f"   [✓] MTER consensus arbitration complete (elapsed: {time.time() - t_stage:.2f}s)")

    # -------------------------------------------------------------------------
    # 9. FINAL CANDIDATE HIGHLIGHTS
    # -------------------------------------------------------------------------
    t_stage = time.time()
    print("\n9. Final Candidate Highlights")
    if not mter_output.candidates:
        print("   -> No candidate satisfied duration or temporal constraints.")
    else:
        top = mter_output.candidates[0]
        sm = top.support_metrics
        print(f"   - Selected Highlight Interval: [{top.proposed_start:.2f}s -> {top.proposed_end:.2f}s]")
        print(f"   - Duration:                    {top.duration:.2f}s")
        print(f"   - Composite Evidence Strength: {top.confidence_estimate:.3f} / 1.000")
        print(f"   - Temporal Agreement IoU:      {top.temporal_agreement_iou:.3f}")
        print(f"   - Modality Support Breakdown:")
        print(f"     * Spoken Transcript: {sm.transcript_support * 100:.1f}% (IoU: {sm.transcript_iou:.2f})")
        print(f"     * Dialogue Dynamics: {sm.conversation_support * 100:.1f}% (IoU: {sm.conversation_iou:.2f})")
        print(f"     * Visual Activity:   {sm.visual_support * 100:.1f}% (IoU: {sm.visual_iou:.2f})")
        print(f"     * Acoustic Prosody:  {sm.prosody_support * 100:.1f}% (IoU: {sm.prosody_iou:.2f})")
        print(f"   - Decision Summary:")
        print(f"     \"{top.decision_summary}\"")
    print(f"   [✓] Candidate highlights extracted (elapsed: {time.time() - t_stage:.2f}s)")

    # -------------------------------------------------------------------------
    # 10. EVALUATION (ONLY WHEN REFERENCE ANNOTATIONS ARE AVAILABLE)
    # -------------------------------------------------------------------------
    t_stage = time.time()
    print("\n10. Evaluation")
    eval_info: Optional[Dict[str, Any]] = None

    if reference_path:
        ref_p = pathlib.Path(reference_path).resolve()
        if ref_p.exists():
            dataset = load_benchmark_dataset(str(ref_p))
            # Match current video to sample in dataset
            matching_sample = None
            for s in dataset.samples:
                if s.video_path and pathlib.Path(s.video_path).name == v_path.name:
                    matching_sample = s
                    break
            if not matching_sample and dataset.samples:
                matching_sample = dataset.samples[0]

            if matching_sample and matching_sample.reference_spans and mter_output.candidates:
                top_span = EvaluationSpan(
                    span_id="mter_top_candidate",
                    start_time=top.proposed_start,
                    end_time=top.proposed_end,
                    duration=top.duration,
                    confidence=top.confidence_estimate,
                    mode_name="mter_full",
                )
                matched, un_preds, un_refs = match_spans_hungarian(
                    predictions=[top_span],
                    references=matching_sample.reference_spans,
                    min_iou=0.10,
                )

                eval_info = {
                    "sample_type": matching_sample.sample_type,
                    "annotation_source": matching_sample.reference_spans[0].annotation_source,
                    "reference_spans": [r.model_dump() for r in matching_sample.reference_spans],
                    "matched_metrics": [m.model_dump() for m in matched],
                    "unmatched_predictions": un_preds,
                    "unmatched_references": un_refs,
                }

                # Save evaluation summary
                with open(out_p / "evaluation_summary.json", "w", encoding="utf-8") as f:
                    json.dump(eval_info, f, indent=2)

                print(f"   - Benchmark Category: {matching_sample.sample_type.upper()}")
                if matching_sample.sample_type == "development":
                    print("     (Development/debug reference annotation used for code verification; NOT research ground truth)")
                
                if matched:
                    res = matched[0]
                    print(f"   - Matched Reference:   [{res.ref_start:.2f}s -> {res.ref_end:.2f}s]")
                    print(f"   - Temporal IoU:        {res.temporal_iou:.3f}")
                    print(f"   - Start Boundary Error:{res.start_error_sec:.2f}s (bias: {res.signed_start_bias_sec:+.2f}s)")
                    print(f"   - End Boundary Error:  {res.end_error_sec:.2f}s (bias: {res.signed_end_bias_sec:+.2f}s)")
                    print(f"   - Duration Error:      {res.duration_error_sec:.2f}s (relative: {res.rel_duration_error * 100:.1f}%)")
                    print(f"   - Hit @ 1.0s / 2.0s:   {res.hit_at_1s} / {res.hit_at_2s}")
                else:
                    print("   - No match: Prediction did not meet the minimum IoU threshold (0.10) against reference.")
        else:
            print(f"   - Reference file not found: {reference_path}")
    else:
        print("   - Prediction-only demonstration (no reference annotations provided; metrics not computed)")

    print(f"   [✓] Evaluation stage complete (elapsed: {time.time() - t_stage:.2f}s)")

    # -------------------------------------------------------------------------
    # HTML REPORT GENERATION
    # -------------------------------------------------------------------------
    html_report_path = out_p / "demo_report.html"
    generate_html_report(
        video_path=str(v_path),
        video_size_mb=v_size_mb,
        total_duration=total_duration,
        extraction_summary=extraction_summary,
        expert_bundles=expert_bundles,
        mter_output=mter_output,
        mter_ledger=mter_ledger,
        evaluation_info=eval_info,
        output_html_path=html_report_path,
    )

    # -------------------------------------------------------------------------
    # DEMO COMPLETE BANNER
    # -------------------------------------------------------------------------
    total_elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"Total Execution Time: {total_elapsed:.2f}s")
    print(f"Artifacts:\n  {out_p}")
    print(f"Report:\n  {html_report_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="ClipSense AI — End-to-End Demonstration Runner (W1-W6)"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file (.mp4, .mov, etc.)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save all demonstration artifacts and the HTML report.",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Optional path to reference annotations JSON (e.g. evaluation/datasets/dev_fixture.json).",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        default=True,
        help="Reuse pre-computed extraction and expert artifacts if present.",
    )
    parser.add_argument(
        "--no-cache",
        dest="use_cache",
        action="store_false",
        help="Force re-extraction and expert evaluation.",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Use deterministic mock LLM for offline testing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Computation device for WhisperX.",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="tiny.en",
        help="Whisper model size.",
    )
    parser.add_argument(
        "--transcript-window-sec",
        type=float,
        default=None,
        help="Duration of transcript analysis window in seconds (e.g. 300-600s to reduce API calls for long videos).",
    )

    args = parser.parse_args()

    run_demonstration(
        video_path=args.video,
        output_dir=args.output_dir,
        reference_path=args.reference,
        use_cache=args.use_cache,
        device=args.device,
        whisper_model=args.whisper_model,
        mock_llm=args.mock_llm,
        transcript_window_sec=args.transcript_window_sec,
    )


if __name__ == "__main__":
    main()
