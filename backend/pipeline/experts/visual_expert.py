"""
ClipSense Visual Expert (W4)

Evaluates timestamped video frame representations and scene transitions to detect
independent visual temporal evidence:
- Visual activity and temporal frame-to-frame change
- Shot / scene boundary transition density
- Significant visual shifts in frame content
- Temporally localized visual event candidate intervals

Temporal Grounding:
- Bounds correspond directly to sampled frame timestamps or scene boundaries.
- Uses SourceResolutionType.FRAME_TIMESTAMP with alignment anchor (e.g. 'frame_15:42').
- Preserves continuous floating-point timestamps: 0.0 <= start_time < end_time <= video_duration.
- Emits ExpertEvidenceBundle containing independent TemporalProposal objects.

Contract & Isolation:
- Strictly independent modality: consumes ONLY visual extraction data (frames.json / VisualData).
- Does NOT ingest transcript, conversation, or acoustic prosody data.
- Does NOT perform cross-modal fusion or MTER aggregation.
- Pluggable visual representation interface (VisualFeatureExtractorInterface).
"""

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.config import VisualConfig, config as app_config
from core.schemas import (
    ExpertEvidenceBundle,
    ProposalSourceMetadata,
    SampledFrameMetadata,
    SceneBoundary,
    SourceResolutionType,
    TemporalProposal,
    VisualData,
)
from pipeline.experts.base import BaseExpert, VisualFeatureExtractorInterface

logger = logging.getLogger("clipsense.experts.visual")


class LightweightFrameDiffExtractor(VisualFeatureExtractorInterface):
    """
    Default pluggable lightweight visual feature extractor.
    
    Extracts spatial intensity representations by downsampling frames to 32x32 grayscale
    and computes normalized L1 distance between consecutive frame representations.
    
    Provides deterministic, fast, reproducible visual activity metrics without requiring
    heavy deep learning models at import or inference time.
    """

    def __init__(self, target_size: Tuple[int, int] = (32, 32)):
        self.target_size = target_size
        self._feature_cache: Dict[str, np.ndarray] = {}

    def extract_features(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Extract a 1D normalized intensity feature vector (1024-d) from a BGR frame.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return np.zeros(self.target_size[0] * self.target_size[1], dtype=np.float32)

        if len(frame_bgr.shape) == 3 and frame_bgr.shape[2] == 3:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        elif len(frame_bgr.shape) == 2:
            gray = frame_bgr
        else:
            gray = frame_bgr[:, :, 0]

        resized = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_AREA)
        features = (resized.astype(np.float32) / 255.0).flatten()
        return features

    def extract_from_path_or_fallback(
        self,
        file_path: Optional[str],
        timestamp: float,
        scene_index: int,
    ) -> np.ndarray:
        """
        Extracts features from an image on disk if available, with deterministic fallback.
        """
        if file_path and file_path in self._feature_cache:
            return self._feature_cache[file_path]

        if file_path and os.path.exists(file_path):
            img = cv2.imread(file_path)
            if img is not None:
                feat = self.extract_features(img)
                self._feature_cache[file_path] = feat
                return feat

        # Deterministic synthetic representation for mock / unit-test environments without disk images
        cache_key = f"synthetic_{timestamp:.3f}_{scene_index}"
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]

        # Use deterministic hash-based sinusoidal pattern keyed on timestamp and scene
        dim = self.target_size[0] * self.target_size[1]
        x = np.linspace(0.0, 2.0 * math.pi, dim, dtype=np.float32)
        freq = 1.0 + (scene_index % 5) * 0.5
        phase = float((timestamp * 1.5) % (2.0 * math.pi))
        feat = (0.5 + 0.5 * np.sin(freq * x + phase)).astype(np.float32)
        self._feature_cache[cache_key] = feat
        return feat

    def compute_distance(self, feat_a: np.ndarray, feat_b: np.ndarray) -> float:
        """
        Compute normalized visual distance (mean absolute difference) in [0.0, 1.0].
        """
        if feat_a is None or feat_b is None:
            return 0.0
        if feat_a.shape != feat_b.shape or feat_a.size == 0:
            return 0.0

        diff = np.abs(feat_a - feat_b)
        dist = float(np.mean(diff))
        return max(0.0, min(1.0, dist))


class VisualExpert(BaseExpert):
    """
    Independent Evidence Expert evaluating visual motion, scene transitions, and visual shifts.
    
    Input:
        VisualData (frames.json, sampled frames metadata, scene boundaries, duration)
    
    Output:
        ExpertEvidenceBundle containing timestamped TemporalProposal objects.
    """

    def __init__(
        self,
        config: Optional[VisualConfig] = None,
        feature_extractor: Optional[VisualFeatureExtractorInterface] = None,
    ):
        super().__init__(name="visual", config=config or app_config.visual)
        self.config: VisualConfig = config or app_config.visual
        self.feature_extractor: VisualFeatureExtractorInterface = (
            feature_extractor or LightweightFrameDiffExtractor()
        )
        self.min_duration: float = self.config.min_candidate_duration_sec
        self.max_duration: float = self.config.max_candidate_duration_sec
        self.smoothing_sec: float = self.config.activity_smoothing_sec
        self.min_percentile: float = self.config.min_activity_percentile

    def evaluate(
        self,
        extraction_data: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExpertEvidenceBundle:
        """
        Evaluates visual frame representations and scene changes to generate
        independent temporal evidence proposals.
        """
        # Validate or parse input to VisualData schema
        if isinstance(extraction_data, dict):
            visual_data = VisualData.model_validate(extraction_data)
        elif isinstance(extraction_data, VisualData):
            visual_data = extraction_data
        else:
            raise TypeError(f"VisualExpert expects VisualData or dict, got {type(extraction_data)}")

        total_duration = float(visual_data.duration)
        if total_duration <= 0.0 and context and "video_duration" in context:
            total_duration = float(context["video_duration"])

        sampled_frames: List[SampledFrameMetadata] = visual_data.sampled_frames
        scene_boundaries: List[SceneBoundary] = visual_data.scene_boundaries

        # Handle empty or insufficient frame input
        if not sampled_frames or len(sampled_frames) < 2:
            logger.warning("Visual Expert received insufficient frames (need >= 2 frames).")
            return ExpertEvidenceBundle(
                expert_name="visual",
                proposals=[],
                dense_features_summary={
                    "total_sampled_frames": len(sampled_frames),
                    "total_scenes_detected": len(scene_boundaries),
                    "status": "insufficient_frames",
                },
            )

        # 1. Extract visual features for each frame
        frame_features: List[np.ndarray] = []
        for f_meta in sampled_frames:
            if hasattr(self.feature_extractor, "extract_from_path_or_fallback"):
                feat = self.feature_extractor.extract_from_path_or_fallback(
                    file_path=f_meta.file_path,
                    timestamp=f_meta.timestamp,
                    scene_index=f_meta.scene_index,
                )
            else:
                # Fallback to direct interface
                img = cv2.imread(f_meta.file_path) if f_meta.file_path and os.path.exists(f_meta.file_path) else None
                feat = self.feature_extractor.extract_features(img)
            frame_features.append(feat)

        # 2. Compute inter-frame visual change metrics
        num_frames = len(sampled_frames)
        raw_diffs = np.zeros(num_frames, dtype=np.float32)
        for i in range(1, num_frames):
            dist = self.feature_extractor.compute_distance(frame_features[i - 1], frame_features[i])
            raw_diffs[i] = dist
        if num_frames > 1:
            raw_diffs[0] = raw_diffs[1]

        # 3. Scene transition density and indicator per frame
        # Identify transition timestamps
        scene_cut_times: List[float] = []
        for sb in scene_boundaries:
            if sb.scene_index > 0:
                scene_cut_times.append(float(sb.start_time))

        frame_timestamps = np.array([float(f.timestamp) for f in sampled_frames], dtype=np.float32)

        # Scene cut bonus: boost frame activity if within 1.5s of a detected cut
        scene_boost = np.zeros(num_frames, dtype=np.float32)
        for cut_t in scene_cut_times:
            near_indices = np.where(np.abs(frame_timestamps - cut_t) <= 1.5)[0]
            scene_boost[near_indices] += 0.25

        combined_activity = raw_diffs + scene_boost

        # 4. Temporal smoothing of visual activity curve
        smoothed_activity = np.zeros(num_frames, dtype=np.float32)
        half_window = self.smoothing_sec / 2.0
        for i, t in enumerate(frame_timestamps):
            in_window = np.where(np.abs(frame_timestamps - t) <= half_window)[0]
            smoothed_activity[i] = float(np.mean(combined_activity[in_window]))

        # 5. Detect candidate visual activity regions
        # Apply configurable percentile threshold
        act_thresh = float(np.percentile(smoothed_activity, self.min_percentile))
        active_mask = smoothed_activity >= act_thresh

        # Cluster contiguous or nearby active frames (merge gap <= 3.0s)
        merge_gap_sec = 3.0
        active_indices = np.where(active_mask)[0]

        candidate_spans: List[Tuple[int, int]] = []
        if len(active_indices) > 0:
            cluster_start = active_indices[0]
            cluster_end = active_indices[0]

            for idx in active_indices[1:]:
                gap = frame_timestamps[idx] - frame_timestamps[cluster_end]
                if gap <= merge_gap_sec:
                    cluster_end = idx
                else:
                    candidate_spans.append((cluster_start, cluster_end))
                    cluster_start = idx
                    cluster_end = idx
            candidate_spans.append((cluster_start, cluster_end))

        # Also consider intervals with high scene transition density if not already covered
        for cut_t in scene_cut_times:
            # Find closest frame index
            cut_idx = int(np.argmin(np.abs(frame_timestamps - cut_t)))
            # Check if covered by existing spans
            covered = any(s <= cut_idx <= e for s, e in candidate_spans)
            if not covered:
                # Add local span around the cut
                left_idx = int(np.argmin(np.abs(frame_timestamps - max(0.0, cut_t - 2.0))))
                right_idx = int(np.argmin(np.abs(frame_timestamps - min(total_duration, cut_t + 2.0))))
                candidate_spans.append((left_idx, right_idx))

        # Sort candidate spans chronologically
        candidate_spans.sort(key=lambda span: frame_timestamps[span[0]])

        # Merge overlapping or close candidate spans
        merged_spans: List[Tuple[int, int]] = []
        for span in candidate_spans:
            if not merged_spans:
                merged_spans.append(span)
                continue
            prev_start, prev_end = merged_spans[-1]
            curr_start, curr_end = span
            if frame_timestamps[curr_start] - frame_timestamps[prev_end] <= merge_gap_sec:
                merged_spans[-1] = (prev_start, max(prev_end, curr_end))
            else:
                merged_spans.append(span)

        # 6. Apply candidate duration bounds and construct TemporalProposals
        proposals: List[TemporalProposal] = []
        prop_counter = 1

        for start_idx, end_idx in merged_spans:
            raw_start = float(frame_timestamps[start_idx])
            raw_end = float(frame_timestamps[end_idx])
            span_dur = raw_end - raw_start

            # If duration is shorter than min_duration, expand symmetrically around the peak
            if span_dur < self.min_duration:
                # Find peak activity frame within span
                span_slice = smoothed_activity[start_idx : end_idx + 1]
                peak_local_idx = int(np.argmax(span_slice)) if len(span_slice) > 0 else 0
                peak_frame_idx = start_idx + peak_local_idx
                peak_time = float(frame_timestamps[peak_frame_idx])

                half_target = self.min_duration / 2.0
                expanded_start = max(0.0, peak_time - half_target)
                expanded_end = min(total_duration, expanded_start + self.min_duration)
                if (expanded_end - expanded_start) < self.min_duration and expanded_start > 0.0:
                    expanded_start = max(0.0, expanded_end - self.min_duration)

                # Snap to closest frame timestamps to maintain exact sensor grounding
                start_idx = int(np.argmin(np.abs(frame_timestamps - expanded_start)))
                end_idx = int(np.argmin(np.abs(frame_timestamps - expanded_end)))

            # If duration is longer than max_duration, trim to highest-activity window of max_duration
            raw_start = float(frame_timestamps[start_idx])
            raw_end = float(frame_timestamps[end_idx])
            span_dur = raw_end - raw_start

            if span_dur > self.max_duration:
                # Find the window of length max_duration with maximum average activity
                best_avg = -1.0
                best_sub = (start_idx, end_idx)
                for s_i in range(start_idx, end_idx + 1):
                    target_end_t = frame_timestamps[s_i] + self.max_duration
                    if target_end_t > total_duration:
                        break
                    e_i = int(np.argmin(np.abs(frame_timestamps - target_end_t)))
                    if e_i > s_i:
                        avg_act = float(np.mean(smoothed_activity[s_i : e_i + 1]))
                        if avg_act > best_avg:
                            best_avg = avg_act
                            best_sub = (s_i, e_i)
                start_idx, end_idx = best_sub

            # Programmatic boundaries
            start_time = float(frame_timestamps[start_idx])
            end_time = float(frame_timestamps[end_idx])
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
                    f"Discarding span [{start_time:.2f}s -> {end_time:.2f}s] outside candidate duration bounds "
                    f"({duration:.2f}s not in [{self.min_duration:.1f}s, {self.max_duration:.1f}s])"
                )
                continue

            # Reject unjustified whole-input proposal
            if total_duration > self.max_duration and duration >= 0.90 * total_duration:
                logger.warning(
                    f"Rejecting whole-input visual proposal spanning {duration:.2f}s of {total_duration:.2f}s"
                )
                continue

            # Compute supporting quantitative features
            span_slice_diffs = raw_diffs[start_idx : end_idx + 1]
            span_slice_act = smoothed_activity[start_idx : end_idx + 1]
            mean_change = float(np.mean(span_slice_diffs)) if len(span_slice_diffs) > 0 else 0.0
            peak_change = float(np.max(span_slice_diffs)) if len(span_slice_diffs) > 0 else 0.0

            # Count scene cuts falling within this candidate interval
            cuts_in_span = [
                cut_t for cut_t in scene_cut_times if start_time <= cut_t <= end_time
            ]
            scene_cut_count = len(cuts_in_span)
            span_frame_count = end_idx - start_idx + 1

            # Evidence-strength estimate (relative non-probabilistic confidence in [0.0, 1.0])
            # Based on relative visual change magnitude and transition activity
            norm_change = min(1.0, mean_change / max(0.15, float(np.mean(raw_diffs) * 1.5)))
            transition_boost = min(0.3, scene_cut_count * 0.1)
            confidence = float(min(1.0, max(0.1, 0.5 * norm_change + 0.3 * (peak_change / 0.5) + transition_boost)))
            confidence = round(confidence, 2)

            # Determine measurable evidence type
            if scene_cut_count >= 2:
                evidence_type = "scene_transition_density"
                explanation = (
                    f"A cluster of {scene_cut_count} scene transitions occurs within this interval "
                    f"with sustained visual changes (mean change score: {mean_change:.3f})."
                )
            elif peak_change >= self.config.motion_change_threshold:
                evidence_type = "visual_shift"
                explanation = (
                    f"A pronounced visual shift occurs in this interval (peak change: {peak_change:.3f}) "
                    f"with {scene_cut_count} scene transition(s)."
                )
            else:
                evidence_type = "visual_activity"
                explanation = (
                    f"A sustained increase in visual activity occurs around this interval "
                    f"(mean change score: {mean_change:.3f}, frames: {span_frame_count})."
                )

            # Distinguish sampled-frame interval from native video FPS
            native_fps = float(visual_data.fps) if visual_data.fps > 0.0 else 25.0
            if len(frame_timestamps) > 1:
                sampling_interval_sec = float(np.median(np.diff(frame_timestamps)))
            else:
                sampling_interval_sec = 1.0

            sampled_start = int(start_idx)
            sampled_end = int(end_idx)
            native_start = int(sampled_frames[sampled_start].frame_index)
            native_end = int(sampled_frames[sampled_end].frame_index)

            proposal = TemporalProposal(
                proposal_id=f"visual_prop_{prop_counter}",
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                confidence_estimate=confidence,
                evidence_type=evidence_type,
                explanation=explanation,
                supporting_features={
                    "visual_change_score": float(round(mean_change, 4)),
                    "peak_visual_change": float(round(peak_change, 4)),
                    "mean_visual_change": float(round(mean_change, 4)),
                    "scene_change_count": int(scene_cut_count),
                    "frame_count": int(span_frame_count),
                    "start_sampled_frame_index": sampled_start,
                    "end_sampled_frame_index": sampled_end,
                    "start_native_frame_index": native_start,
                    "end_native_frame_index": native_end,
                    "sampling_interval_sec": float(round(sampling_interval_sec, 4)),
                    "native_fps": float(native_fps),
                },
                source_metadata=ProposalSourceMetadata(
                    source_type=SourceResolutionType.FRAME_TIMESTAMP,
                    temporal_resolution_sec=float(sampling_interval_sec),
                    alignment_anchor=f"sampled_frame_{sampled_start}:{sampled_end} (native_frame_{native_start}:{native_end})",
                    sample_rate_or_fps=float(native_fps),
                ),
            )
            proposals.append(proposal)
            prop_counter += 1

        logger.info(f"Visual Expert produced {len(proposals)} localized temporal proposals.")

        if len(frame_timestamps) > 1:
            global_sampling_interval = float(np.median(np.diff(frame_timestamps)))
        else:
            global_sampling_interval = 1.0
        global_native_fps = float(visual_data.fps) if visual_data.fps > 0.0 else 25.0

        return ExpertEvidenceBundle(
            expert_name="visual",
            proposals=proposals,
            dense_features_summary={
                "total_sampled_frames": int(num_frames),
                "sampling_interval_sec": float(round(global_sampling_interval, 4)),
                "native_fps": float(global_native_fps),
                "total_scenes_detected": int(len(scene_boundaries)),
                "mean_visual_change": float(np.mean(raw_diffs)),
                "max_visual_change": float(np.max(raw_diffs)),
                "activity_threshold": float(act_thresh),
                "encoder": str(self.config.active_encoder),
            },
        )
