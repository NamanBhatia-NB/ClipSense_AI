# B.TECH MINOR PROJECT INTERIM PROGRESS REPORT
**Academic Session 2026–2027 | Semester VII**

---

# CLIPSENSE AI: MULTIMODAL TEMPORAL EVIDENCE REASONING (MTER) FOR HIGH-VALUE SHORT-FORM VIDEO HIGHLIGHT EXTRACTION

<br>

**Submitted in partial fulfillment of the requirements for the award of the degree of**  
**Bachelor of Technology in Computer Science & Engineering**

<br>

### Submitted By:
**Naman Bhatia**  
B.Tech CSE, 7th Semester  
Department of Computer Science & Engineering  

### Under the Supervision of:
**Project Supervisor / Faculty Mentor**  
Department of Computer Science & Engineering  

---

<div style="page-break-after: always;"></div>

## ABSTRACT

The rapid proliferation of long-form video content—such as technical lectures, podcasts, panel interviews, and keynote presentations—has created an urgent demand for automated, high-fidelity short-form clip generation. Conventional automated video clippers predominantly depend on naive heuristics: they either submit ungrounded, monolithic transcripts to Large Language Models (LLMs) or trigger highlights based solely on transient acoustic energy spikes. These approaches suffer from severe limitations, including unimodal hallucination, context loss ("lost-in-the-middle" attention deficit), temporal boundary drift, and premature score collapse. 

To overcome these deficiencies, this research introduces **ClipSense AI**, an automated system powered by a novel **Multimodal Temporal Evidence Reasoner (MTER)**. The ClipSense framework decouples the detection of salient moments into four domain-specific evidence experts: (1) **Transcript Expert** (semantic claims and topical completeness), (2) **Conversation Expert** (dialogue pacing and discourse closure), (3) **Visual Expert** (scene change density and visual activity), and (4) **Prosody Expert** (pitch variation and acoustic energy dynamics). Instead of early scalar score fusion, evidence proposals maintain continuous floating-point timestamps and are arbitrated within an event-grounded temporal compatibility graph. MTER clusters boundary candidates, resolves multi-modality temporal conflicts, and generates an auditable, inspectable decision ledger. 

To date, Milestones W1 through W6 have been completed and verified. A comprehensive benchmark evaluation suite featuring Hungarian bipartite matching and a dedicated observability pipeline have been established. Across automated testing, the system achieves 100% test passage (67/67 unit and integration tests), demonstrates a temporal Intersection over Union (IoU) of 0.955 against development reference annotations, and processes long-form videos (up to 46.7 minutes) end-to-end. Remaining objectives focus on micro-boundary acoustic snapping (W7), dynamic 9:16 active-speaker video reframing (W8), and broad-scale empirical benchmarking.

---

## 1. RESEARCH GAP & PROBLEM FORMULATION

Existing academic literature and commercial systems in video summarization and highlight clipping exhibit four critical foundational shortcomings:

```
+----------------------------------------------------------------------------------------------------+
|                                    CURRENT INDUSTRY LIMITATIONS                                    |
+------------------------------------+----------------------------------+----------------------------+
| 1. Unimodal LLM Hallucination      | 2. Premature Score Collapse      | 3. Temporal Discretization |
|    - Evaluates text in isolation   |    - Collapses signals into a    |    - Truncates millisecond |
|    - Ignores vocal emphasis        |      single arbitrary scalar     |      timestamps to integer |
|    - Suffers attention degradation |    - Discards modal provenance   |      seconds, cutting off  |
|      in long transcripts           |    - Masks inter-sensor conflicts|      syllables and words   |
+------------------------------------+----------------------------------+----------------------------+
                                                 |
                                                 v
+----------------------------------------------------------------------------------------------------+
|                                   CLIPSENSE MULTIMODAL PARADIGM                                    |
|   4 Independent Domain Experts -> Continuous Time -> Graph Compatibility -> MTER Auditable Ledger  |
+----------------------------------------------------------------------------------------------------+
```

### 1.1 Unimodal Semantic Bias and Attention Degradation
Most contemporary automated clipping tools extract the spoken transcript via automatic speech recognition (ASR) and pass the text directly to an LLM with instructions to "identify virality." This paradigm treats video as static text, ignoring critical paralinguistic cues (intonation, sarcasm, urgency) and physical visual actions. Furthermore, when long transcripts (exceeding 30 minutes) are ingested in a single context window, transformers exhibit severe **primacy and recency bias** ("lost-in-the-middle"), over-indexing on video introductions and conclusions while ignoring salient middle passages.

### 1.2 Premature Score Collapse vs. Late Evidence Arbitration
Current multimodal highlight detectors commonly merge heterogeneous feature vectors (e.g., audio loudness, optical flow, word embeddings) into a single scalar score early in the pipeline. This *premature score collapse* irrevocably destroys modality provenance: a downstream selector cannot determine whether an interval scored 0.85 due to intense semantic dialogue, loud background noise, or a sudden camera movement. Without preserving distinct evidence vectors, contradictory signals cannot be reconciled.

### 1.3 Temporal Quantization and Syllable Severing
Conventional video processing pipelines discretize continuous time into coarse integer seconds or fixed video-frame intervals (e.g., 1.0-second buckets). This rounding discards sub-second phonetic alignment, causing clips to abruptly cut off speaker syllables, interrupt trailing breath pauses, or slice through mid-sentence clauses.

### 1.4 The "Black Box" Interpretability Deficit
Standard deep learning summarizers emit boundary timestamps without auditability. In academic research and media production, human editors require explainable justifications: *Which modalities agreed on this boundary? What acoustic or linguistic conflict occurred? Why was this start point favored over an adjacent scene cut?* Current systems offer zero transparency regarding arbitration decisions.

---

## 2. PROJECT OBJECTIVES: STATUS & ROADMAP

The ClipSense AI engineering lifecycle is divided into structured weekly milestones (W1 to W8). The section below details the objectives fully completed to date alongside the remaining developmental scope.

```
=====================================================================================================
MILESTONE ROADMAP EXECUTION PROGRESS:
[==================================================W1-W6: COMPLETED=========================>] [W7-W8: PENDING]
W1: Architecture    W2: Multimodal     W3: Semantic       W4: Physical       W5: MTER Reasoning  W6: Evaluation   W7: Micro-Trim   W8: Render Engine
    & Schemas           Extraction         Experts            Experts            & Arbitration       & Benchmark      & Snapping       & Reframing
  [COMPLETE]          [COMPLETE]         [COMPLETE]         [COMPLETE]         [COMPLETE]          [COMPLETE]       [PENDING]        [PENDING]
=====================================================================================================
```

### 2.1 Objectives Achieved So Far (Weeks 1 to 6)

#### Objective 1: Strict Architectural Formalization & Continuous-Time Contracts (Week 1)
* Established clean package architecture (`core/`, `app/`, `pipeline/`, `evaluation/`, `demo/`).
* Formulated immutable Pydantic schemas enforcing continuous 64-bit floating-point timestamps across all sensory representations.
* Implemented `ProposalSourceMetadata` to track physical sensor resolution, hardware source, and temporal alignment anchors, preventing numerical drift across pipeline boundaries.
* Designed the `EvidenceLedger` schema to hold full structural audit trails.

#### Objective 2: Decoupled Multimodal Sensory Extraction Pipeline (Week 2)
* **Linguistic Modality:** Integrated WhisperX coupled with `wav2vec2` forced phonetic alignment in strict execution mode, generating word-level millisecond timestamps without heuristic rounding.
* **Visual Modality:** Engineered dual-layer video analysis combining uniform frame sampling with PySceneDetect content-aware transition detection (`ContentDetector`) to record cut vectors.
* **Prosodic Modality:** Implemented digital signal processing in Librosa to compute physical acoustic metrics: continuous autocorrelation fundamental frequency ($F_0$), root-mean-square (RMS) energy, dynamic range, and voicing fraction.
* **Conversational Structure:** Formulated speaker turn segmentation distinguishing multi-speaker exchanges from single-speaker monologue units using parameterized pause thresholds ($\tau > 0.7\text{s}$).

#### Objective 3: Domain-Specific Modality Evidence Experts (Weeks 3 & 4)
* **Transcript Expert (W3):** Constructed bounded sliding temporal windows to eliminate attention decay; leveraged Google Gemini structured JSON outputs strictly constrained to word-index pairs (`start_word_index`, `end_word_index`), programmatically mapped to WhisperX float timestamps.
* **Conversation Expert (W3):** Built discourse unit segmentation analyzing structural setup, conversational development, and rhetorical payoff while enforcing monologue safety constraints.
* **Visual Expert (W4):** Built lightweight visual difference encoders ($32 \times 32$ normalized $L_1$ frame intensity diffs) and scene-transition density curves, operating 100% locally with zero external LLM dependency.
* **Prosody Expert (W4):** Implemented localized dynamic baseline tracking (30-second sliding windows) to isolate legitimate vocal emphasis spikes ($+15\%$ to $+35\%$ pitch and volume excursions) from steady-state background music or ambient noise.

#### Objective 4: Multimodal Temporal Evidence Reasoner (MTER) Prototype (Week 5)
* **Common Evidence Bundle:** Built unified data structures compiling asynchronous proposals into indexed temporal grids.
* **Temporal Compatibility Graph:** Formulated graph $G = (V, E)$ where vertices represent modality proposals and edges represent temporal overlap and center-distance compatibility ($IoU \ge 0.10$, center gap $\le 12.0\text{s}$). Connected components isolate cohesive **Event Regions**, preventing invalid boundary cross-pairing.
* **Intra-Region Clustering & Conflict Detection:** Implemented density-based clustering with a 3.0-second tolerance window to identify consensus start/end boundaries, accompanied by explicit `TemporalConflict` logging for contradictory signals.
* **Candidate Arbitration:** Evaluated multi-dimensional candidate intervals across duration constraints ($10.0\text{s} \le \Delta t \le 45.0\text{s}$), scoring composite support, boundary agreement tightness, and linguistic closure.

#### Objective 5: Formal Research Baseline & Quantitative Evaluation Framework (Week 6)
* **Legacy Baseline Implementation:** Created an isolated transcript-only baseline mirroring legacy single-modality heuristic workflows.
* **Bipartite Hungarian Matching:** Built strict bipartite graph matching (`match_spans_hungarian`) pairing predicted highlight intervals with ground-truth reference annotations based on temporal IoU maximization.
* **Standardized Metric Suite:** Implemented temporal IoU, boundary error (start, end, duration bias in seconds), precision, recall, and F1 at multiple tolerance thresholds ($\text{Hit}@1.0\text{s}$, $\text{Hit}@2.0\text{s}$).
* **Dataset Management:** Established clean dataset schemas with explicit separation between development fixtures and research benchmark datasets.

#### Objective 6: Live Demonstration, Observability & Adaptive Scaling (Week 6)
* Created a standalone demonstration runner (`demo/run_demo.py`) surfacing all 10 pipeline stages with live telemetry, counts, and duration tracking.
* Built automated, self-contained visual HTML report generation (`demo_report.html`) featuring interactive CSS timeline bars for all 4 modality proposal layers and MTER selected candidates.
* Implemented **Adaptive Temporal Windowing** for long-form media: dynamically scales LLM analysis windows for long videos ($\approx \text{duration} / 5.5$), reducing API round-trips from 63 down to 6 calls (a 90% reduction) to operate safely within API rate limits.
* Validated the complete system on both a 90-second benchmark conversational sample and a 46.7-minute long-form technical video.

---

### 2.2 Summary of Experimental Validation & Quantitative Results

| Evaluation Dimension | Parameter / Metric | Measured Empirical Result | Scientific Significance |
| :--- | :--- | :--- | :--- |
| **System Reliability** | Automated Test Suite | **67 / 67 Tests Passed (100%)** | Zero regressions across schemas, extraction, experts, MTER, and evaluation. |
| **Benchmark Alignment** | Temporal IoU (90s Dev Fixture) | **0.955 (95.5% overlap)** | Confirms tight alignment with human reference interval `[0.09s -> 27.27s]`. |
| **Boundary Precision** | Start Boundary Error | **0.06 seconds** | Validates word-level phonetic alignment precision. |
| **Boundary Precision** | End Boundary Error | **1.16 seconds** | Snaps accurately to speaker pause boundary without cutting speech. |
| **Temporal Hit Rate** | Boundary Tolerance ($\text{Hit}@2.0\text{s}$) | **True (Hit)** | Satisfies high-precision video editing criteria ($< 2.0\text{s}$ tolerance). |
| **Scalability Stress Test** | Video Duration / Proposals | **46.7 min / 491 Proposals** | Successfully generated 113 event regions, arbitrating 960 candidate intervals. |

---

### 2.3 Remaining Objectives & Future Roadmap (Weeks 7 and 8)

The remaining research and development tasks are strictly bounded to post-arbitration refinement and vertical media rendering:

```
+----------------------------------------------------------------------------------------------------+
|                                    REMAINING DEVELOPMENT SCOPE                                     |
+----------------------------------------------------+-----------------------------------------------+
| Week 7: Temporal Micro-Boundary Refinement         | Week 8: Active Speaker Reframing & Rendering  |
| - Local micro-trimming (+/- 1.5s window)           | - LR-ASD (Lightweight Active Speaker Detect)  |
| - Zero-crossing audio waveform cut alignment       | - Smooth kinematic camera framing (9:16)      |
| - Syntactic clause & breath pause snapping         | - Dynamic subtitle burning (.ass / FFmpeg)    |
| - Dead-air elimination                             | - Production containerization & batch exports |
+----------------------------------------------------+-----------------------------------------------+
```

#### Objective 7: Acoustic/Linguistic Micro-Boundary Refinement (Week 7)
* **Waveform Zero-Crossing Alignment:** Implement local waveform analysis within a $\pm 1.5$-second window around MTER cut boundaries to ensure audio splits occur at precise zero-amplitude crossings, eliminating audible digital clicks and pops.
* **Linguistic Breath Pause Snapping:** Align start cuts to speech onset and end cuts to sentence-final breath pauses identified by WhisperX word confidence drops and Librosa energy troughs.
* **Dead-Air Buffer Trimming:** Enforce an automated $0.35$-second silence buffer around spoken words to preserve conversational cadence without lingering dead air.

#### Objective 8: Visual Active Speaker Detection & Dynamic 9:16 Reframing (Week 8)
* **Lightweight Real-time Active Speaker Detection (LR-ASD):** Integrate face bounding box tracking with lip-motion energy correlation to determine the active conversational speaker in multi-speaker video feeds.
* **Kinematic Camera Tracking:** Formulate smooth trajectory filtering (Kalman filter or exponential moving average) to pan and crop horizontal $16:9$ footage into vertical $9:16$ aspect ratio without erratic jumping.
* **Subtitle Rendering Engine:** Automate the generation of word-level animated `.ass` subtitle files from aligned transcript tokens and compile final broadcast-ready `.mp4` video clips via GPU-accelerated FFmpeg.

---

## 3. SUMMARY OF CURRENT TECHNICAL ACHIEVEMENTS

1. **Frozen Architecture Integrity:** The foundational core established in Weeks 1–5 remains modular, mathematically rigorous, and frozen. The Week 6 deliverables were implemented purely as non-invasive evaluation and demonstration layers.
2. **Empirical Grounding:** The pipeline has been validated across both short representative samples and long-form continuous video, confirming robust handling of dense multimodal proposals (up to 491 proposals and 113 event regions).
3. **Observability:** Complete auditability has been realized through `EvidenceLedger` JSON serializations and interactive HTML reports, fulfilling the goal of transparent, explainable AI highlight selection.

---

## 4. REFERENCES

1. **Bain, M., Huh, J., Han, T., & Zisserman, A.** (2020). *WhisperX: Time-Accurate Speech Recognition of Long-Form Audio with Word-Level Timestamps.* In *Proc. INTERSPEECH 2023*, pp. 4883–4887.
2. **Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I.** (2023). *Robust Speech Recognition via Large-Scale Weak Supervision.* In *International Conference on Machine Learning (ICML)*, PMLR, pp. 28492–28518.
3. **McFee, B., Raffel, C., Liang, D., Ellis, D. P., McVicar, M., Battenberg, E., & Nieto, O.** (2015). *librosa: Audio and Music Signal Analysis in Python.* In *Proceedings of the 14th Python in Science Conference*, pp. 18–25.
4. **Castellano, G., Kessous, L., & Caridakis, G.** (2008). *Emotion Recognition through Multiple Modalities: Face, Body Gesture, Speech.* In *Affect and Emotion in Human-Computer Interaction*, Springer, Berlin, Heidelberg, pp. 92–103.
5. **Kuhn, H. W.** (1955). *The Hungarian Method for the Assignment Problem.* *Naval Research Logistics Quarterly*, 2(1‐2), 83–97.
6. **Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P.** (2024). *Lost in the Middle: How Language Models Use Long Contexts.* *Transactions of the Association for Computational Linguistics (TACL)*, 12, 157–173.
7. **Alcázar, J., Caba, F., Long, C., & Ghanem, B.** (2020). *Active Speakers in Context.* In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 12465–12474.
8. **Ester, M., Kriegel, H. P., Sander, J., & Xu, X.** (1996). *A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise.* In *KDD*, Vol. 96, No. 34, pp. 226–240.
9. **Gygli, M., Grabner, H., Riemenschneider, H., & Van Gool, L.** (2014). *Creating Summaries from User Videos.* In *European Conference on Computer Vision (ECCV)*, Springer, Cham, pp. 505–520.
10. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I.** (2017). *Attention Is All You Need.* In *Advances in Neural Information Processing Systems (NeurIPS)*, 30, pp. 5998–6008.
