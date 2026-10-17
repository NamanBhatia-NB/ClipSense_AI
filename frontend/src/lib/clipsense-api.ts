/**
 * ClipSense AI - Local API Client (Week 9)
 *
 * Provides typed methods for interacting with the local FastAPI backend:
 * - Health and readiness monitoring
 * - Video upload and processing triggering
 * - Task status polling and progress monitoring
 * - Highlight candidate and modality evidence inspection
 * - Stream and download URL resolution for exported MP4 clips
 */

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  ffmpeg_available: boolean;
  device: string;
  timestamp: number;
}

export interface TaskStatusResponse {
  task_id: string;
  status: "queued" | "processing" | "completed" | "failed";
  stage: "queued" | "extraction" | "experts" | "bundle" | "mter" | "rendering" | "completed" | "failed";
  progress: number;
  message: string;
  error?: string | null;
  created_at: number;
  completed_at?: number | null;
}

export interface ModalityEvidenceSummary {
  transcript_support: number;
  conversation_support: number;
  visual_support: number;
  prosody_support: number;
  linguistic_boundary_support: number;
  strong_modalities: string[];
  moderate_modalities: string[];
  weak_modalities: string[];
}

export interface HighlightClipInfo {
  filename: string;
  stream_url: string;
  download_url: string;
  file_size_bytes?: number | null;
  status: "success" | "failed" | "skipped";
}

export interface BoundaryRefinementSummary {
  refined: boolean;
  original_start: number;
  original_end: number;
  original_duration: number;
  refined_start: number;
  refined_end: number;
  refined_duration: number;
  start_delta_sec: number;
  end_delta_sec: number;
  start_evidence_type?: string | null;
  end_evidence_type?: string | null;
  signals_used: string[];
  rationale: string;
  fallback_reason?: string | null;
}

export interface HighlightItem {
  candidate_id: string;
  rank: number;
  start_time: number;
  end_time: number;
  duration: number;
  confidence_estimate: number;
  temporal_agreement_iou: number;
  decision_summary: string;
  modality_evidence: ModalityEvidenceSummary;
  clip?: HighlightClipInfo | null;
  boundary_refinement?: BoundaryRefinementSummary | null;
}

export interface TaskHighlightsResponse {
  task_id: string;
  status: "queued" | "processing" | "completed" | "failed";
  video_filename: string;
  total_duration: number;
  total_candidates: number;
  highlights: HighlightItem[];
  execution_time_sec?: number | null;
}

export function getApiBaseUrl(): string {
  if (typeof window !== "undefined" && process.env.NEXT_PUBLIC_CLIPSENSE_API_URL) {
    return process.env.NEXT_PUBLIC_CLIPSENSE_API_URL;
  }
  return "http://localhost:8000";
}

/**
 * Check backend readiness and FFmpeg availability.
 */
export async function checkBackendHealth(): Promise<{ available: boolean; ffmpeg: boolean; data?: HealthResponse; error?: string }> {
  const baseUrl = getApiBaseUrl();
  try {
    const res = await fetch(`${baseUrl}/api/health`, {
      method: "GET",
      headers: { Accept: "application/json" },
      cache: "no-store",
    });
    if (!res.ok) {
      return { available: false, ffmpeg: false, error: `Backend responded with HTTP ${res.status}` };
    }
    const data = (await res.json()) as HealthResponse;
    return { available: true, ffmpeg: data.ffmpeg_available, data };
  } catch (err) {
    return {
      available: false,
      ffmpeg: false,
      error: err instanceof Error ? err.message : "Failed to connect to local API bridge at " + baseUrl,
    };
  }
}

/**
 * Submit an uploaded video file for processing.
 */
export async function submitVideoUpload(
  file: File,
  options?: { top_k?: number; use_cache?: boolean; render_mode?: string; refine_boundaries?: boolean }
): Promise<{ task_id: string; message: string; video_filename: string }> {
  const baseUrl = getApiBaseUrl();
  const formData = new FormData();
  formData.append("file", file);
  formData.append("top_k", String(options?.top_k ?? 3));
  formData.append("use_cache", String(options?.use_cache ?? true));
  formData.append("render_mode", options?.render_mode ?? "accurate");
  formData.append("refine_boundaries", String(options?.refine_boundaries ?? true));

  const res = await fetch(`${baseUrl}/api/process`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error(errData.detail || `Upload failed with status ${res.status}`);
  }

  return (await res.json()) as { task_id: string; message: string; video_filename: string };
}

/**
 * Submit a local video path reference for processing.
 */
export async function submitVideoPath(
  videoPath: string,
  options?: { top_k?: number; use_cache?: boolean; render_mode?: string; refine_boundaries?: boolean }
): Promise<{ task_id: string; message: string; video_filename: string }> {
  const baseUrl = getApiBaseUrl();
  const formData = new FormData();
  formData.append("video_path", videoPath);
  formData.append("top_k", String(options?.top_k ?? 3));
  formData.append("use_cache", String(options?.use_cache ?? true));
  formData.append("render_mode", options?.render_mode ?? "accurate");
  formData.append("refine_boundaries", String(options?.refine_boundaries ?? true));

  const res = await fetch(`${baseUrl}/api/process`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error(errData.detail || `Submission failed with status ${res.status}`);
  }

  return (await res.json()) as { task_id: string; message: string; video_filename: string };
}

/**
 * Retrieve active status for a processing task.
 */
export async function getTaskStatus(taskId: string): Promise<TaskStatusResponse> {
  const baseUrl = getApiBaseUrl();
  const res = await fetch(`${baseUrl}/api/status/${encodeURIComponent(taskId)}`, {
    method: "GET",
    headers: { Accept: "application/json" },
    cache: "no-store",
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error(errData.detail || `Failed to retrieve task status (${res.status})`);
  }

  return (await res.json()) as TaskStatusResponse;
}

/**
 * Retrieve final ranked highlights and evidence metrics for a task.
 */
export async function getTaskHighlights(taskId: string): Promise<TaskHighlightsResponse> {
  const baseUrl = getApiBaseUrl();
  const res = await fetch(`${baseUrl}/api/highlights/${encodeURIComponent(taskId)}`, {
    method: "GET",
    headers: { Accept: "application/json" },
    cache: "no-store",
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error(errData.detail || `Failed to retrieve highlights (${res.status})`);
  }

  return (await res.json()) as TaskHighlightsResponse;
}

/**
 * Resolve full playback stream URL for an exported clip.
 */
export function getClipStreamUrl(taskId: string, filename: string): string {
  const baseUrl = getApiBaseUrl();
  return `${baseUrl}/api/clips/${encodeURIComponent(taskId)}/${encodeURIComponent(filename)}`;
}

/**
 * Resolve direct download URL for an exported clip.
 */
export function getClipDownloadUrl(taskId: string, filename: string): string {
  const baseUrl = getApiBaseUrl();
  return `${baseUrl}/api/clips/${encodeURIComponent(taskId)}/${encodeURIComponent(filename)}?download=true`;
}
