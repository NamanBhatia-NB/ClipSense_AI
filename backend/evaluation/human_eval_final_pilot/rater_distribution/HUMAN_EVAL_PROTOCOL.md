# ClipSense AI — Human Evaluation Protocol & Rating Guide

> [!NOTE]
> **Study Status**: This package represents a prepared pilot protocol and blinded evaluation material; data collection and ratings have not been conducted. No human-rated quality results are claimed.

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
2. **System B (Simple Multimodal Aggregation)**: Non-MTER multimodal baseline pooling proposals across available modalities, ranked by unweighted average modality coverage ($S_{agg} = \frac{1}{M} \sum_m \text{coverage}_m$) with greedy NMS (0.30).
3. **System C (MTER)**: Multimodal Temporal Evidence Reasoner performing consensus arbitration via compatibility graph, event clustering, boundary clustering, and heuristic composite evidence scoring ($S_{composite} \in [0, 1]$).
4. **System D (MTER Refined)**: Directly corresponds to the exact base candidate selected by System C, applying speech/pause boundary snapping ($\Delta s = -0.15$s, $\Delta e = +0.15$s). This ensures a clean ablation isolating the boundary refinement effect.

> **Control Separation**: A uniform sliding-window control baseline (regular 30s windows every 15s with fixed nominal 0.850 confidence) is maintained separately to demonstrate non-semantic windowing behavior. It is excluded from the primary 4-system human rating pool.

> **Pilot Study Scope**: This evaluation protocol covers 5 sub-hour source videos (20 total clips, 60 planned rating assignments across 3 raters) as a prepared pilot protocol for evaluating inter-rater agreement, qualitative engagement, and boundary perception. Ratings have not yet been collected.

---

## Statistical Analysis Plan
1. **Unit of Analysis**: Ratings are averaged across raters per clip before system-level aggregation.
2. **Repeated Measures**: Linear Mixed-Effects Models (LMM) with `source_video_id` and `rater_id` as random intercepts.
3. **C-vs-D Ablation**: Exact paired Wilcoxon signed-rank test on matched base candidates.
4. **Inter-Rater Reliability**: Krippendorff's Alpha and ICC(2,k).
