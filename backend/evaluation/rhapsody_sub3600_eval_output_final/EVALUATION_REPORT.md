# ClipSense AI — Sub-Hour Rhapsody Replay-Engagement & Timestamp Evaluation Report (Corrected)

## 1. Benchmark Metadata & Sub-Hour Verification

All evaluated source videos are official test episodes from the **Rhapsody** benchmark (Park et al., COLM 2025).
Every video was verified against the strict **sub-hour constraint (duration <= 3,600s)**.

| Sample ID | YouTube ID | Actual Title | Channel / Source | Duration (s) | Duration (min) | Status |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| `rhapsody_real_r0VyLTLGcWk` | `r0VyLTLGcWk` | Getting In Bed w/ Mia Malkova & Gabby Epstein \| First Date with Lauren Compton | YMH Studios | 2779.0s | 46m 19s | PASSED (< 3,600s) |
| `rhapsody_real_hSuSutvGdXQ` | `hSuSutvGdXQ` | Fox News MELTS DOWN Live on AIR as Trump LOSES IT | MeidasTouch | 1120.0s | 18m 40s | PASSED (< 3,600s) |
| `rhapsody_real_h91NpMIgbdI` | `h91NpMIgbdI` | 'Gutless'! Walmart Tosses DEI After Right-Wing Conservatives TARGET Public Companies | Roland Martin | Roland S. Martin | 1165.0s | 19m 25s | PASSED (< 3,600s) |
| `rhapsody_real_brYEgU782YU` | `brYEgU782YU` | NJ Mystery Drones Situation is Crazy, Congressman Claims Secret Iran Mothership Off US Coast, & More | Philip DeFranco | 1451.0s | 24m 11s | PASSED (< 3,600s) |
| `rhapsody_real_gEcW5rta3wU` | `gEcW5rta3wU` | Two Wigs, a Gun and a Murder Part 2 \| Full Episode | 48 Hours | 2408.0s | 40m 08s | PASSED (< 3,600s) |

> [!IMPORTANT]
> **Correction of Previous Metadata**: Previous informal summaries incorrectly referred to these videos as Lex Fridman or Huberman Lab podcasts.
> The actual dataset manifests verify that the 5 sub-hour test episodes are YMH Studios, MeidasTouch, Roland S. Martin, Philip DeFranco, and 48 Hours episodes.

## 2. System Baseline Definitions

### Primary Experimental Systems (A, B, C, D)

| System | Category | Description & Provenance | Modalities | Selection Score Provenance |
| :--- | :--- | :--- | :--- | :--- |
| **System A: Transcript-Only Semantic Baseline** | Transcript Lexical / Semantic Importance Proposals | Spoken transcript proposals from TranscriptExpert, scored by lexical/semantic importance (confidence in [0.58, 0.81]), filtered by greedy NMS (0.30). | `transcript` | Pure inference score (zero reference leakage) |
| **System B: Simple Multimodal Aggregation Baseline** | Multimodal Unweighted Coverage Sum | Non-MTER multimodal baseline pooling proposals across transcript, conversation, visual, and prosody. Scored by unweighted average modality coverage, greedy NMS (0.30). | `transcript, conversation, visual, prosody` | Pure inference score (zero reference leakage) |
| **System C: MTER Multi-Modal Reasoner** | Multimodal Temporal Evidence Consensus Reasoner | Consensus arbitration across 4 modalities using compatibility graph, event clustering, boundary clustering, and heuristic composite evidence scoring. | `transcript, conversation, visual, prosody` | Pure inference score (zero reference leakage) |
| **System D: MTER + Boundary Refinement (Production Pipeline)** | MTER + Evidence-Aware Acoustic & Syntactic Refinement | Production pipeline combining MTER consensus arbitration with sub-second acoustic pause alignment and syntactic sentence boundary snapping. | `transcript, conversation, visual, prosody` | Pure inference score (zero reference leakage) |

### Control System (Uniform Sliding Window)

| System | Category | Description & Provenance | Modalities | Selection Score Provenance |
| :--- | :--- | :--- | :--- | :--- |
| **Control: Uniform Sliding Window Baseline** | Structural Non-Semantic Sliding Windows | Control baseline generating regular 30s temporal windows every 15s with constant nominal score 0.850. Retained to demonstrate non-semantic coverage behavior. | `none` | Fixed nominal 0.850 / chronological order |

### Stored Evidence Modality Audit & Proposal Availability

| Episode ID | Transcript Proposals | Conversation Proposals | Visual Proposals | Prosody Proposals | Active Modal Evidence |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `rhapsody_real_brYEgU782YU` | 59 | 1 | 0 (EMPTY) | 0 (EMPTY) | Transcript + Conversation |
| `rhapsody_real_gEcW5rta3wU` | 109 | 42 | 0 (EMPTY) | 0 (EMPTY) | Transcript + Conversation |
| `rhapsody_real_h91NpMIgbdI` | 46 | 4 | 0 (EMPTY) | 0 (EMPTY) | Transcript + Conversation |
| `rhapsody_real_hSuSutvGdXQ` | 50 | 18 | 0 (EMPTY) | 0 (EMPTY) | Transcript + Conversation |
| `rhapsody_real_r0VyLTLGcWk` | 187 | 44 | 0 (EMPTY) | 0 (EMPTY) | Transcript + Conversation |

> [!WARNING]
> **Modality Availability Disclosure**: The Rhapsody benchmark (`younghanstark/rhapsody`) distributes official WhisperX transcript word timestamps and YouTube replay graphs.
> Full-length raw `.mp4` video files and uncompressed audio waveforms were not distributed in the benchmark dataset files.
> Consequently, `visual_evidence.json` and `prosody_evidence.json` exist but are **empty (0 proposals)** across all 5 episodes.
> Systems B, C, and D operated over the available spoken language evidence (transcript semantic units + conversational thematic turns), rather than a full 4-modality audiovisual stream. System B averages strictly over available modalities with proposal evidence.

> [!NOTE]
> **System A Resolution**: We distinguish genuine **System A (Transcript-Only Semantic Baseline)** from **Control (Uniform Sliding Window)**.
> System A (Semantic) ranks genuine `TranscriptExpert` proposals using spoken semantic importance scores (confidence in [0.58, 0.81]) with greedy NMS (0.30).
> The uniform sliding window baseline is isolated as an explicit control to demonstrate why a constant-score windowing heuristic is NOT a semantic detector.

## 3. Adapted 100-Bin Replay-Engagement Evaluation Results

Evaluated using the adapted COLM 2025 Rhapsody protocol: 100 equal-duration temporal segments per episode, segment-0 intro click bias corrected, ground-truth replay threshold > 0.50 (max 20 segments).
- **Segment Ranking (mAP)**: Evaluated across the complete ranked 100-bin predicted curve.
- **Binary Metrics (Hit Rate, Precision, Recall, F1)**: Evaluated at standard `segment_prediction_k = 20` segments independently of clip count.
- **Union Coverage**: True disjoint interval union (`union_coverage_pct`), strictly bounded in [0%, 100%]. Overlapping duration sum reported separately as `summed_clip_duration_ratio_pct`.

### Common Budget: Exactly K=1 Clip

#### Primary Systems (A, B, C, D)

| System | mAP | Hit Rate | Precision | Recall | F1 | Actual Clips (Mean [Min, Max]) | Mean Union Duration (s) | True Union Coverage (%) | Summed Ratio (%) | Intra-Overlap Rate | Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **System A: Transcript-Only Semantic Baseline** | **0.1137** (±0.132) | 0.4000 | 0.0300 | 0.1667 | 0.0507 | 1.0 [1, 1] | 29.4s | **2.1%** | 2.1% | 0.0000 | 0.020s |
| **System B: Simple Multimodal Aggregation Baseline** | **0.0478** (±0.024) | 0.2000 | 0.0200 | 0.1000 | 0.0333 | 1.0 [1, 1] | 22.3s | **1.3%** | 1.3% | 0.0000 | 0.069s |
| **System C: MTER Multi-Modal Reasoner** | **0.0475** (±0.025) | 0.2000 | 0.0200 | 0.1000 | 0.0333 | 1.0 [1, 1] | 33.7s | **1.9%** | 1.9% | 0.0000 | 0.474s |
| **System D: MTER + Boundary Refinement (Production Pipeline)** | **0.0475** (±0.025) | 0.2000 | 0.0200 | 0.1000 | 0.0333 | 1.0 [1, 1] | 34.0s | **1.9%** | 1.9% | 0.0000 | 0.575s |

#### Control System

| System | mAP | Hit Rate | Precision | Recall | F1 | Actual Clips (Mean [Min, Max]) | Mean Union Duration (s) | True Union Coverage (%) | Summed Ratio (%) | Intra-Overlap Rate | Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Control: Uniform Sliding Window Baseline** | 0.0490 (±0.024) | 0.4000 | 0.0300 | 0.1667 | 0.0507 | 1.0 [1, 1] | 30.0s | 1.9% | 1.9% | 0.0000 | 0.001s |

### Budget: At Most K=3 Clips

#### Primary Systems (A, B, C, D)

| System | mAP | Hit Rate | Precision | Recall | F1 | Actual Clips (Mean [Min, Max]) | Mean Union Duration (s) | True Union Coverage (%) | Summed Ratio (%) | Intra-Overlap Rate | Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **System A: Transcript-Only Semantic Baseline** | **0.1265** (±0.124) | 0.4000 | 0.0300 | 0.2667 | 0.0538 | 3.0 [3, 3] | 125.7s | **8.3%** | 8.5% | 0.0079 | 0.020s |
| **System B: Simple Multimodal Aggregation Baseline** | **0.0766** (±0.056) | 0.4000 | 0.0300 | 0.2000 | 0.0515 | 3.0 [3, 3] | 75.8s | **4.9%** | 5.3% | 0.0381 | 0.069s |
| **System C: MTER Multi-Modal Reasoner** | **0.0601** (±0.040) | 0.2000 | 0.0100 | 0.0500 | 0.0167 | 3.0 [3, 3] | 108.0s | **6.2%** | 6.2% | 0.0000 | 0.474s |
| **System D: MTER + Boundary Refinement (Production Pipeline)** | **0.0601** (±0.040) | 0.2000 | 0.0100 | 0.0500 | 0.0167 | 3.0 [3, 3] | 108.9s | **6.3%** | 6.3% | 0.0000 | 0.575s |

#### Control System

| System | mAP | Hit Rate | Precision | Recall | F1 | Actual Clips (Mean [Min, Max]) | Mean Union Duration (s) | True Union Coverage (%) | Summed Ratio (%) | Intra-Overlap Rate | Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Control: Uniform Sliding Window Baseline** | 0.0490 (±0.024) | 0.4000 | 0.0300 | 0.1667 | 0.0507 | 3.0 [3, 3] | 60.0s | 3.9% | 5.8% | 0.2222 | 0.001s |

### Budget: At Most K=5 Clips

#### Primary Systems (A, B, C, D)

| System | mAP | Hit Rate | Precision | Recall | F1 | Actual Clips (Mean [Min, Max]) | Mean Union Duration (s) | True Union Coverage (%) | Summed Ratio (%) | Intra-Overlap Rate | Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **System A: Transcript-Only Semantic Baseline** | **0.1157** (±0.126) | 0.4000 | 0.0300 | 0.2667 | 0.0538 | 5.0 [5, 5] | 228.4s | **14.9%** | 15.5% | 0.0083 | 0.020s |
| **System B: Simple Multimodal Aggregation Baseline** | **0.0916** (±0.089) | 0.4000 | 0.0300 | 0.2500 | 0.0530 | 5.0 [5, 5] | 129.9s | **8.5%** | 9.4% | 0.0258 | 0.069s |
| **System C: MTER Multi-Modal Reasoner** | **0.1015** (±0.066) | 0.6000 | 0.0400 | 0.3500 | 0.0712 | 5.0 [5, 5] | 175.1s | **10.2%** | 10.2% | 0.0000 | 0.474s |
| **System D: MTER + Boundary Refinement (Production Pipeline)** | **0.1015** (±0.066) | 0.6000 | 0.0400 | 0.3500 | 0.0712 | 5.0 [5, 5] | 176.3s | **10.3%** | 10.3% | 0.0000 | 0.575s |

#### Control System

| System | mAP | Hit Rate | Precision | Recall | F1 | Actual Clips (Mean [Min, Max]) | Mean Union Duration (s) | True Union Coverage (%) | Summed Ratio (%) | Intra-Overlap Rate | Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Control: Uniform Sliding Window Baseline** | 0.0490 (±0.024) | 0.4000 | 0.0300 | 0.1667 | 0.0507 | 5.0 [5, 5] | 90.0s | 5.8% | 9.7% | 0.1333 | 0.001s |

### Budget: At Most K=20 Clips

#### Primary Systems (A, B, C, D)

| System | mAP | Hit Rate | Precision | Recall | F1 | Actual Clips (Mean [Min, Max]) | Mean Union Duration (s) | True Union Coverage (%) | Summed Ratio (%) | Intra-Overlap Rate | Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **System A: Transcript-Only Semantic Baseline** | **0.1273** (±0.124) | 0.4000 | 0.0300 | 0.3000 | 0.0545 | 20.0 [20, 20] | 924.0s | **58.6%** | 69.2% | 0.0088 | 0.020s |
| **System B: Simple Multimodal Aggregation Baseline** | **0.0694** (±0.048) | 0.4000 | 0.0300 | 0.2000 | 0.0515 | 20.0 [20, 20] | 612.5s | **40.7%** | 47.5% | 0.0080 | 0.069s |
| **System C: MTER Multi-Modal Reasoner** | **0.0741** (±0.039) | 0.8000 | 0.0400 | 0.3000 | 0.0697 | 20.0 [20, 20] | 609.0s | **37.3%** | 37.3% | 0.0000 | 0.474s |
| **System D: MTER + Boundary Refinement (Production Pipeline)** | **0.0738** (±0.039) | 0.8000 | 0.0400 | 0.3000 | 0.0697 | 20.0 [20, 20] | 614.1s | **37.6%** | 37.7% | 0.0000 | 0.575s |

#### Control System

| System | mAP | Hit Rate | Precision | Recall | F1 | Actual Clips (Mean [Min, Max]) | Mean Union Duration (s) | True Union Coverage (%) | Summed Ratio (%) | Intra-Overlap Rate | Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Control: Uniform Sliding Window Baseline** | 0.0490 (±0.024) | 0.4000 | 0.0300 | 0.1667 | 0.0507 | 20.0 [20, 20] | 315.0s | 20.3% | 38.6% | 0.0333 | 0.001s |
### All Candidates Returned by Each Configured System

> [!NOTE]
> **Candidate Pool Budgeting Disclosure**: Systems A, B, and Control return their full unconstrained candidate pools (averaging 39.0, 59.0, and 118.4 clips per episode). MTER Systems C and D were evaluated with their default configured budget cap (`top_k=20`), returning exactly 20 candidates per episode.

#### Primary Systems (A, B, C, D)

| System | mAP | Hit Rate | Precision | Recall | F1 | Actual Clips (Mean [Min, Max]) | Mean Union Duration (s) | True Union Coverage (%) | Summed Ratio (%) | Intra-Overlap Rate | Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **System A: Transcript-Only Semantic Baseline** | **0.1190** (±0.126) | 0.4000 | 0.0300 | 0.3000 | 0.0545 | 39.0 [24, 63] | 1681.5s | **93.6%** | 118.9% | 0.0076 | 0.020s |
| **System B: Simple Multimodal Aggregation Baseline** | **0.0683** (±0.048) | 0.4000 | 0.0300 | 0.2000 | 0.0515 | 59.0 [29, 107] | 1633.1s | **91.7%** | 114.0% | 0.0051 | 0.069s |
| **System C: MTER Multi-Modal Reasoner** | **0.0741** (±0.039) | 0.8000 | 0.0400 | 0.3000 | 0.0697 | 20.0 [20, 20] | 609.0s | **37.3%** | 37.3% | 0.0000 | 0.474s |
| **System D: MTER + Boundary Refinement (Production Pipeline)** | **0.0738** (±0.039) | 0.8000 | 0.0400 | 0.3000 | 0.0697 | 20.0 [20, 20] | 614.1s | **37.6%** | 37.7% | 0.0000 | 0.575s |

#### Control System

| System | mAP | Hit Rate | Precision | Recall | F1 | Actual Clips (Mean [Min, Max]) | Mean Union Duration (s) | True Union Coverage (%) | Summed Ratio (%) | Intra-Overlap Rate | Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Control: Uniform Sliding Window Baseline** | 0.0436 (±0.026) | 0.2000 | 0.0200 | 0.1000 | 0.0333 | 118.4 [74, 185] | 1784.6s | 100.0% | 198.4% | 0.0065 | 0.001s |

## 4. Key Scientific Insights & Limitations

1. **Five-Episode Rhapsody Test Subset Limitation**:
   - These results evaluate a five-episode sub-hour test subset from `yhpark/rhapsody` (COLM 2025). They characterize system behavior under query-free long-form speech but should not be claimed as full-benchmark generalization across all 13,364 dataset episodes.
2. **Replay-Engagement Alignment vs Human Editorial Cuts**:
   - Rhapsody 100-bin annotations represent aggregated YouTube viewer replay intensity across 11.2s to 27.8s temporal buckets. They are NOT exact human editorial cuts.
   - Differences between candidate boundaries and Rhapsody bin edges reflect viewer replaying patterns rather than editorial clipping errors.
3. **Boundary Refinement & Alignment Characterization**:
   - **Candidate-Dependent Refinement**: Boundary shifts in System D are candidate-dependent rather than uniform. Specific adjustments depend on the temporal proximity of raw candidate boundaries to sentence onsets/offsets, speech activity, and shot boundaries.
   - **Distinct Alignment Mechanisms**:
     - *Sentence-Boundary Snapping*: Snaps start and end boundaries to spoken sentence completions derived from WhisperX word alignments. Across the evaluated Rhapsody test candidates, this produced start shifts such as −0.15s (onset padding), +0.42s, +0.871s, and +0.931s (snapping forward to sentence onset + padding).
     - *Configured Padding*: A fixed configurable buffer (default $\pm 0.15$s) applied around detected sentence boundaries to prevent clipped speech onsets or trailing word cutoffs.
     - *Visual-Scene-Cut Alignment*: In the 90-second audiovisual demonstration where visual frames were processed, snapping to nearby shot transitions produced a +1.313s adjustment to avoid abrupt mid-shot cuts.
     - *Acoustic-Pause Confirmation*: Energy-based silence detection via `librosa`/`soundfile` is fully implemented and verified under unit tests (`tests/test_boundary_refiner.py`). However, acoustic-pause confirmation was **not triggered or exercised in the stored Rhapsody benchmark runs or audiovisual end-to-end demonstrations**; boundary shifts in stored runs were driven strictly by sentence boundaries and visual scene cuts.
   - **Replay Alignment Impact**: Because 100-bin Rhapsody segments span 11s–28s each, sub-second boundary shifts maintain identical macro-bin replay alignment (IoU > 0.96 between C and D). We make no claims of human-perceived boundary quality improvement prior to collecting actual human rater responses.
4. **Zero Reference Leakage Safeguard**:
   - Every candidate in `predictions_timestamp_only.csv` was selected strictly by its inference-time heuristic ranking score.
   - Ground-truth replay labels, reference highlight spans, and Hungarian match IoU were completely excluded from candidate selection.
