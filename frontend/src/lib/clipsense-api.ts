/**
 * ClipSense - Local API Client
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
  stage: "queued" | "extraction" | "experts" | "bundle" | "mter" | "refinement" | "rendering" | "completed" | "failed";
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

export function normalizeApiBaseUrl(rawUrl: string): string {
  const url = rawUrl.trim().replace(/\/+$/, "");

  // If user provided Hugging Face Space repository URL instead of direct app domain:
  // e.g. https://huggingface.co/spaces/NamanBhatia-NB/ClipSense_AI -> https://namanbhatia-nb-clipsense-ai.hf.space
  const hfPattern = /^https?:\/\/huggingface\.co\/spaces\/([a-zA-Z0-9_.-]+)\/([a-zA-Z0-9_.-]+)/i;
  const hfSpaceMatch = hfPattern.exec(url);
  if (hfSpaceMatch) {
    const user = (hfSpaceMatch[1] ?? "").toLowerCase().replace(/_/g, "-");
    const space = (hfSpaceMatch[2] ?? "").toLowerCase().replace(/_/g, "-");
    return `https://${user}-${space}.hf.space`;
  }

  return url;
}

export function getApiBaseUrl(): string {
  const envUrl = process.env.NEXT_PUBLIC_CLIPSENSE_API_URL;
  if (envUrl && envUrl.trim().length > 0) {
    return normalizeApiBaseUrl(envUrl);
  }
  return "http://localhost:8000";
}

interface ApiErrorDetail {
  detail?: string;
}

async function extractApiErrorMessage(res: Response, fallback: string): Promise<string> {
  try {
    const errData = (await res.json()) as ApiErrorDetail;
    if (typeof errData?.detail === "string" && errData.detail.trim().length > 0) {
      return errData.detail;
    }
  } catch {
    // Non-JSON or unreachable body fallback
  }
  return fallback;
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

export interface UserClipItem {
  clip_id: string;
  task_id: string;
  filename: string;
  stream_url: string;
  download_url: string;
  source_video: string;
  start_time: number;
  end_time: number;
  duration: number;
  file_size_bytes?: number | null;
  created_at: number;
  rank?: number;
  aspect_ratio?: string;
  subtitles?: boolean;
  score?: number;
  mter_event_id?: string;
}

export interface UserClipsResponse {
  user_id: string;
  total_clips: number;
  clips: UserClipItem[];
}

/**
 * Submit an uploaded video file for processing.
 */
export async function submitVideoUpload(
  file: File,
  options?: {
    top_k?: number;
    use_cache?: boolean;
    render_mode?: string;
    refine_boundaries?: boolean;
    aspect_ratio?: string;
    add_subtitles?: boolean;
    user_id?: string;
  }
): Promise<{ task_id: string; message: string; video_filename: string }> {
  const baseUrl = getApiBaseUrl();
  const formData = new FormData();
  formData.append("file", file);
  formData.append("top_k", String(options?.top_k ?? 3));
  formData.append("use_cache", String(options?.use_cache ?? true));
  formData.append("render_mode", options?.render_mode ?? "accurate");
  formData.append("refine_boundaries", String(options?.refine_boundaries ?? true));
  formData.append("aspect_ratio", options?.aspect_ratio ?? "9:16");
  formData.append("add_subtitles", String(options?.add_subtitles ?? true));
  if (options?.user_id) {
    formData.append("user_id", options.user_id);
  }

  const res = await fetch(`${baseUrl}/api/process`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errorMsg = await extractApiErrorMessage(res, `Upload failed with status ${res.status}`);
    throw new Error(errorMsg);
  }

  return (await res.json()) as { task_id: string; message: string; video_filename: string };
}

/**
 * Submit a local video path reference for processing.
 */
export async function submitVideoPath(
  videoPath: string,
  options?: {
    top_k?: number;
    use_cache?: boolean;
    render_mode?: string;
    refine_boundaries?: boolean;
    aspect_ratio?: string;
    add_subtitles?: boolean;
    user_id?: string;
  }
): Promise<{ task_id: string; message: string; video_filename: string }> {
  const baseUrl = getApiBaseUrl();
  const formData = new FormData();
  formData.append("video_path", videoPath);
  formData.append("top_k", String(options?.top_k ?? 3));
  formData.append("use_cache", String(options?.use_cache ?? true));
  formData.append("render_mode", options?.render_mode ?? "accurate");
  formData.append("refine_boundaries", String(options?.refine_boundaries ?? true));
  formData.append("aspect_ratio", options?.aspect_ratio ?? "9:16");
  formData.append("add_subtitles", String(options?.add_subtitles ?? true));
  if (options?.user_id) {
    formData.append("user_id", options.user_id);
  }

  const res = await fetch(`${baseUrl}/api/process`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errorMsg = await extractApiErrorMessage(res, `Submission failed with status ${res.status}`);
    throw new Error(errorMsg);
  }

  return (await res.json()) as { task_id: string; message: string; video_filename: string };
}

function resolveFullApiUrl(pathOrUrl?: string | null): string {
  if (!pathOrUrl || typeof pathOrUrl !== "string") return "";
  const trimmed = pathOrUrl.trim();
  if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
    return trimmed;
  }
  const baseUrl = getApiBaseUrl();
  const cleanPath = trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
  return `${baseUrl}${cleanPath}`;
}

/**
 * Retrieve all local clips generated for a user.
 */
export async function getUserClips(userId: string): Promise<UserClipsResponse> {
  const baseUrl = getApiBaseUrl();
  const res = await fetch(`${baseUrl}/api/users/${encodeURIComponent(userId)}/clips`, {
    method: "GET",
    headers: { Accept: "application/json" },
    cache: "no-store",
  });

  if (!res.ok) {
    const errorMsg = await extractApiErrorMessage(res, `Failed to fetch user clips (${res.status})`);
    throw new Error(errorMsg);
  }

  const data = (await res.json()) as UserClipsResponse;
  return {
    ...data,
    clips: (data.clips ?? []).map((clip) => ({
      ...clip,
      stream_url: resolveFullApiUrl(clip.stream_url),
      download_url: resolveFullApiUrl(clip.download_url),
    })),
  };
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
    const errorMsg = await extractApiErrorMessage(res, `Failed to retrieve task status (${res.status})`);
    throw new Error(errorMsg);
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
    const errorMsg = await extractApiErrorMessage(res, `Failed to retrieve highlights (${res.status})`);
    throw new Error(errorMsg);
  }

  const data = (await res.json()) as TaskHighlightsResponse;
  return {
    ...data,
    highlights: (data.highlights ?? []).map((h) => ({
      ...h,
      clip: h.clip
        ? {
            ...h.clip,
            stream_url: resolveFullApiUrl(h.clip.stream_url),
            download_url: resolveFullApiUrl(h.clip.download_url),
          }
        : null,
    })),
  };
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
