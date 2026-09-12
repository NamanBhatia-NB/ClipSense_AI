"""
ClipSense Prosody Expert (W4)

Evaluates extracted physical acoustic features to detect independent prosodic temporal evidence:
- Vocal pitch (F0) dynamics and excursions
- RMS energy / vocal loudness elevation
- Voicing fraction filtering (silent/unvoiced discrimination)
- Dynamic moving local acoustic baseline (e.g. 30s centered window)
- Localized vocal emphasis candidate intervals

Temporal Grounding:
- Bounds correspond directly to extracted prosody window timestamps.
- Uses SourceResolutionType.PROSODY_WINDOW with alignment anchor (e.g. 'window_24:58').
- Preserves continuous floating-point timestamps: 0.0 <= start_time < end_time <= video_duration.
- Emits ExpertEvidenceBundle containing independent TemporalProposal objects.

Contract & Isolation:
- Strictly independent modality: consumes ONLY acoustic prosody extraction data (prosody.json / ProsodyData).
- Does NOT ingest transcript, conversation, or visual data.
- Does NOT perform cross-modal fusion or MTER aggregation.
- No LLM for raw acoustic measurement; no direct emotion classification.
- Configurable baselines and thresholds via ProsodyConfig (no hard-coded magic numbers).
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.config import ProsodyConfig, config as app_config
from core.schemas import (
    ExpertEvidenceBundle,
    ProposalSourceMetadata,
    ProsodyData,
    ProsodyWindow,
    SourceResolutionType,
    TemporalProposal,
)
from pipeline.experts.base import BaseExpert

logger = logging.getLogger("clipsense.experts.prosody")


class ProsodyExpert(BaseExpert):
    """
    Independent Evidence Expert evaluating acoustic pitch excursions and vocal energy.
    
    Input:
        ProsodyData (prosody.json, windowed F0, RMS energy, voicing fraction, duration)
    
    Output:
        ExpertEvidenceBundle containing timestamped TemporalProposal objects.
    """

    def __init__(self, config: Optional[ProsodyConfig] = None):
        super().__init__(name="prosody", config=config or app_config.prosody)
        self.config: ProsodyConfig = config or app_config.prosody
        self.baseline_window_sec: float = self.config.local_baseline_window_sec
        self.min_duration: float = self.config.min_candidate_duration_sec
        self.max_duration: float = self.config.max_candidate_duration_sec
        self.merge_gap_sec: float = self.config.merge_gap_sec
        self.min_voicing_fraction: float = self.config.min_voicing_fraction
        self.f0_weight: float = self.config.emphasis_f0_weight
        self.energy_weight: float = self.config.emphasis_energy_weight
        self.threshold_mult: float = self.config.emphasis_threshold_multiplier

    def evaluate(
        self,
        extraction_data: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExpertEvidenceBundle:
        """
        Evaluates physical acoustic features to generate independent temporal evidence proposals.
        """
        # Validate or parse input to ProsodyData schema
        if isinstance(extraction_data, dict):
            prosody_data = ProsodyData.model_validate(extraction_data)
        elif isinstance(extraction_data, ProsodyData):
            prosody_data = extraction_data
        else:
            raise TypeError(f"ProsodyExpert expects ProsodyData or dict, got {type(extraction_data)}")

        total_duration = float(prosody_data.duration)
        if total_duration <= 0.0 and context and "video_duration" in context:
            total_duration = float(context["video_duration"])

        windows: List[ProsodyWindow] = prosody_data.windows

        # Handle empty or insufficient windows
        if not windows or len(windows) < 2:
            logger.warning("Prosody Expert received insufficient acoustic windows.")
            return ExpertEvidenceBundle(
                expert_name="prosody",
                proposals=[],
                dense_features_summary={
                    "total_windows_analyzed": len(windows),
                    "status": "insufficient_windows",
                },
            )

        num_windows = len(windows)
        window_starts = np.array([float(w.start_time) for w in windows], dtype=np.float64)
        window_ends = np.array([float(w.end_time) for w in windows], dtype=np.float64)
        window_centers = (window_starts + window_ends) / 2.0

        mean_f0s = np.array([float(w.mean_f0) for w in windows], dtype=np.float64)
        rms_energies = np.array([float(w.rms_energy) for w in windows], dtype=np.float64)
        voicing_fractions = np.array([float(w.voicing_fraction) for w in windows], dtype=np.float64)

        # 1. Voiced speech mask: filter out silent / unvoiced windows from baseline calculation
        voiced_mask = (voicing_fractions >= self.min_voicing_fraction) & (mean_f0s > 0.0)

        # Compute global voiced baselines as fallback
        if np.any(voiced_mask):
            global_median_f0 = float(np.median(mean_f0s[voiced_mask]))
            global_median_energy = float(np.median(rms_energies[voiced_mask]))
        else:
            global_median_f0 = 150.0  # safe neutral fallback
            global_median_energy = float(np.median(rms_energies)) if len(rms_energies) > 0 else 0.05

        # 2. Compute local moving baseline for F0 and RMS energy
        local_f0_baselines = np.zeros(num_windows, dtype=np.float64)
        local_energy_baselines = np.zeros(num_windows, dtype=np.float64)
        half_baseline_sec = self.baseline_window_sec / 2.0

        for i in range(num_windows):
            t_c = window_centers[i]
            # Identify windows within centered local baseline duration
            in_local_range = np.abs(window_centers - t_c) <= half_baseline_sec
            local_voiced = in_local_range & voiced_mask

            if np.any(local_voiced):
                local_f0_baselines[i] = float(np.median(mean_f0s[local_voiced]))
                local_energy_baselines[i] = float(np.median(rms_energies[local_voiced]))
            else:
                local_f0_baselines[i] = global_median_f0
                local_energy_baselines[i] = global_median_energy

        # 3. Compute relative elevation ratios and composite emphasis score
        f0_elevations = np.ones(num_windows, dtype=np.float64)
        energy_elevations = np.ones(num_windows, dtype=np.float64)
        composite_scores = np.zeros(num_windows, dtype=np.float64)

        for i in range(num_windows):
            if voiced_mask[i]:
                base_f0 = max(10.0, local_f0_baselines[i])
                base_eng = max(1e-5, local_energy_baselines[i])
                f0_elevations[i] = mean_f0s[i] / base_f0
                energy_elevations[i] = rms_energies[i] / base_eng
                composite_scores[i] = (
                    self.f0_weight * f0_elevations[i] + self.energy_weight * energy_elevations[i]
                )
            else:
                # Unvoiced / silence receives no emphasis
                f0_elevations[i] = 1.0
                energy_elevations[i] = 1.0
                composite_scores[i] = 0.0

        # 4. Identify emphasis windows and cluster into candidate intervals
        # Candidate emphasis condition: voiced AND composite score >= threshold multiplier
        emphasis_mask = voiced_mask & (composite_scores >= self.threshold_mult)

        # Fallback if no window strictly exceeds threshold multiplier (e.g. highly flat audio):
        # pick the top 20% highest composite scores among voiced windows
        if not np.any(emphasis_mask) and np.any(voiced_mask):
            voiced_comp = composite_scores[voiced_mask]
            rel_thresh = float(np.percentile(voiced_comp, 80.0))
            emphasis_mask = voiced_mask & (composite_scores >= rel_thresh)

        emphasis_indices = np.where(emphasis_mask)[0]

        candidate_clusters: List[Tuple[int, int]] = []
        if len(emphasis_indices) > 0:
            cluster_start = emphasis_indices[0]
            cluster_end = emphasis_indices[0]

            for idx in emphasis_indices[1:]:
                # Gap between end of previous window and start of current window
                gap = window_starts[idx] - window_ends[cluster_end]
                if gap <= self.merge_gap_sec:
                    cluster_end = idx
                else:
                    candidate_clusters.append((cluster_start, cluster_end))
                    cluster_start = idx
                    cluster_end = idx
            candidate_clusters.append((cluster_start, cluster_end))

        # 5. Apply candidate duration bounds and temporal grounding
        proposals: List[TemporalProposal] = []
        prop_counter = 1

        for c_start, c_end in candidate_clusters:
            raw_start_t = float(window_starts[c_start])
            raw_end_t = float(window_ends[c_end])
            span_dur = raw_end_t - raw_start_t

            # If duration is shorter than min_candidate_duration_sec, symmetrically expand window indices
            if span_dur < self.min_duration:
                # Center around peak emphasis window
                cluster_slice = composite_scores[c_start : c_end + 1]
                peak_local_idx = int(np.argmax(cluster_slice)) if len(cluster_slice) > 0 else 0
                peak_win_idx = c_start + peak_local_idx
                peak_time = float(window_centers[peak_win_idx])

                half_target = self.min_duration / 2.0
                target_start_t = max(0.0, peak_time - half_target)
                target_end_t = min(total_duration, target_start_t + self.min_duration)
                if (target_end_t - target_start_t) < self.min_duration and target_start_t > 0.0:
                    target_start_t = max(0.0, target_end_t - self.min_duration)

                # Snap to closest window boundaries
                c_start = int(np.argmin(np.abs(window_starts - target_start_t)))
                c_end = int(np.argmin(np.abs(window_ends - target_end_t)))

            # If duration is longer than max_candidate_duration_sec, trim to highest-emphasis subsegment
            raw_start_t = float(window_starts[c_start])
            raw_end_t = float(window_ends[c_end])
            span_dur = raw_end_t - raw_start_t

            if span_dur > self.max_duration:
                best_avg = -1.0
                best_sub = (c_start, c_end)
                for s_i in range(c_start, c_end + 1):
                    target_end = window_starts[s_i] + self.max_duration
                    if target_end > total_duration:
                        break
                    e_i = int(np.argmin(np.abs(window_ends - target_end)))
                    if e_i > s_i:
                        avg_comp = float(np.mean(composite_scores[s_i : e_i + 1]))
                        if avg_comp > best_avg:
                            best_avg = avg_comp
                            best_sub = (s_i, e_i)
                c_start, c_end = best_sub

            # Exact timestamps derived from prosody windows
            start_time = float(window_starts[c_start])
            end_time = float(window_ends[c_end])
            duration = max(0.0, end_time - start_time)

            # Strict bounds validation: 0.0 <= start_time < end_time <= total_duration
            if start_time < 0.0 or end_time > total_duration or start_time >= end_time:
                logger.warning(
                    f"Discarding proposal with invalid timestamps [{start_time:.3f}s -> {end_time:.3f}s] "
                    f"vs video duration {total_duration:.2f}s"
                )
                continue

            # Candidate duration validation
            if duration < self.min_duration or duration > self.max_duration:
                logger.warning(
                    f"Discarding candidate span [{start_time:.2f}s -> {end_time:.2f}s] outside duration bounds "
                    f"({duration:.2f}s not in [{self.min_duration:.1f}s, {self.max_duration:.1f}s])"
                )
                continue

            # Reject whole-input proposal
            if total_duration > self.max_duration and duration >= 0.90 * total_duration:
                logger.warning(
                    f"Rejecting whole-input prosody proposal spanning {duration:.2f}s of {total_duration:.2f}s"
                )
                continue

            # Compute supporting quantitative features across this candidate interval
            span_slice_comp = composite_scores[c_start : c_end + 1]
            span_slice_f0 = mean_f0s[c_start : c_end + 1]
            span_slice_eng = rms_energies[c_start : c_end + 1]
            span_slice_f0_elev = f0_elevations[c_start : c_end + 1]
            span_slice_eng_elev = energy_elevations[c_start : c_end + 1]
            span_slice_vf = voicing_fractions[c_start : c_end + 1]

            mean_comp = float(np.mean(span_slice_comp)) if len(span_slice_comp) > 0 else 1.0
            peak_comp = float(np.max(span_slice_comp)) if len(span_slice_comp) > 0 else 1.0
            mean_f0_elev = float(np.mean(span_slice_f0_elev)) if len(span_slice_f0_elev) > 0 else 1.0
            mean_eng_elev = float(np.mean(span_slice_eng_elev)) if len(span_slice_eng_elev) > 0 else 1.0
            mean_f0 = float(np.mean(span_slice_f0)) if len(span_slice_f0) > 0 else 0.0
            mean_energy = float(np.mean(span_slice_eng_elev)) if len(span_slice_eng) > 0 else 0.0
            mean_vf = float(np.mean(span_slice_vf)) if len(span_slice_vf) > 0 else 0.0
            win_count = c_end - c_start + 1

            # Evidence-strength estimate (relative non-probabilistic confidence in [0.0, 1.0])
            # Based on relative elevation above baseline (1.0 = baseline, 1.5 = +50%)
            elev_diff = max(0.0, mean_comp - 1.0)
            confidence = float(min(1.0, max(0.15, 0.45 + 0.55 * (elev_diff / 0.6))))
            confidence = round(confidence, 2)

            # Measurable evidence explanation generated faithfully from actual measured feature values
            f0_pct = (mean_f0_elev - 1.0) * 100.0
            energy_pct = (mean_eng_elev - 1.0) * 100.0

            if f0_pct > 0.5 and energy_pct > 0.5:
                dynamic_desc = (
                    f"Both vocal pitch (+{f0_pct:.1f}%) and acoustic energy (+{energy_pct:.1f}%) increase "
                    f"relative to the local {self.baseline_window_sec:.0f}s acoustic baseline"
                )
            elif f0_pct > 0.5 and energy_pct <= 0.5:
                if energy_pct < -0.5:
                    dynamic_desc = (
                        f"Vocal pitch increases (+{f0_pct:.1f}%) while acoustic energy decreases ({energy_pct:+.1f}%) "
                        f"relative to the local {self.baseline_window_sec:.0f}s acoustic baseline"
                    )
                else:
                    dynamic_desc = (
                        f"Vocal pitch increases (+{f0_pct:.1f}%) with energy near baseline ({energy_pct:+.1f}%) "
                        f"relative to the local {self.baseline_window_sec:.0f}s acoustic baseline"
                    )
            elif energy_pct > 0.5 and f0_pct <= 0.5:
                if f0_pct < -0.5:
                    dynamic_desc = (
                        f"Acoustic energy increases (+{energy_pct:.1f}%) while vocal pitch decreases ({f0_pct:+.1f}%) "
                        f"relative to the local {self.baseline_window_sec:.0f}s acoustic baseline"
                    )
                else:
                    dynamic_desc = (
                        f"Acoustic energy increases (+{energy_pct:.1f}%) with pitch near baseline ({f0_pct:+.1f}%) "
                        f"relative to the local {self.baseline_window_sec:.0f}s acoustic baseline"
                    )
            else:
                dynamic_desc = (
                    f"A transient vocal emphasis peak (peak composite: {peak_comp:.2f}x) occurs in this interval, "
                    f"while span-averaged pitch ({f0_pct:+.1f}%) and energy ({energy_pct:+.1f}%) remain near the local {self.baseline_window_sec:.0f}s baseline"
                )

            explanation = f"{dynamic_desc} (mean voicing fraction: {mean_vf:.2f})."

            hop_sec = float(prosody_data.hop_length_sec) if prosody_data.hop_length_sec > 0.0 else 0.5

            proposal = TemporalProposal(
                proposal_id=f"prosody_prop_{prop_counter}",
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                confidence_estimate=confidence,
                evidence_type="vocal_emphasis",
                explanation=explanation,
                supporting_features={
                    "f0_change": float(round(mean_f0_elev, 4)),
                    "energy_change": float(round(mean_eng_elev, 4)),
                    "composite_emphasis": float(round(mean_comp, 4)),
                    "peak_composite_emphasis": float(round(peak_comp, 4)),
                    "mean_f0_hz": float(round(mean_f0, 2)),
                    "mean_energy": float(round(mean_energy, 4)),
                    "voicing_fraction": float(round(mean_vf, 3)),
                    "window_count": int(win_count),
                    "start_window_index": int(c_start),
                    "end_window_index": int(c_end),
                },
                source_metadata=ProposalSourceMetadata(
                    source_type=SourceResolutionType.PROSODY_WINDOW,
                    temporal_resolution_sec=float(hop_sec),
                    alignment_anchor=f"window_{int(c_start)}:{int(c_end)}",
                    sample_rate_or_fps=float(prosody_data.sample_rate),
                ),
            )
            proposals.append(proposal)
            prop_counter += 1

        logger.info(f"Prosody Expert produced {len(proposals)} localized temporal proposals.")

        return ExpertEvidenceBundle(
            expert_name="prosody",
            proposals=proposals,
            dense_features_summary={
                "total_windows_analyzed": int(num_windows),
                "voiced_windows_count": int(np.sum(voiced_mask)),
                "global_median_f0_hz": float(global_median_f0),
                "global_median_energy": float(global_median_energy),
                "local_baseline_window_sec": float(self.baseline_window_sec),
            },
        )
