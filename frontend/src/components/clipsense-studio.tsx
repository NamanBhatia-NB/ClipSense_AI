"use client";

import React, { useEffect, useState, useRef } from "react";
import {
  AlertCircle,
  CheckCircle2,
  Clock,
  Download,
  Film,
  Loader2,
  Sliders,
  Sparkles,
  UploadCloud,
  Volume2,
  Users,
  Eye,
  FileText,
  RefreshCw,
} from "lucide-react";
import { toast } from "sonner";
import {
  checkBackendHealth,
  getClipDownloadUrl,
  getClipStreamUrl,
  getTaskHighlights,
  getTaskStatus,
  submitVideoUpload,
  type HighlightItem,
  type TaskHighlightsResponse,
  type TaskStatusResponse,
} from "~/lib/clipsense-api";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Input } from "./ui/input";
import { useSession } from "next-auth/react";
import { Label } from "./ui/label";

interface ClipSenseStudioProps {
  onProcessingComplete?: () => void;
}

export function ClipSenseStudio({ onProcessingComplete }: ClipSenseStudioProps = {}) {
  const { data: session } = useSession();

  const onProcessingCompleteRef = useRef(onProcessingComplete);
  useEffect(() => {
    onProcessingCompleteRef.current = onProcessingComplete;
  }, [onProcessingComplete]);

  // Backend connection state
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);

  // File upload state (Local file upload only)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Configuration options: Number of clips strictly between 1 and 10
  const [topK, setTopK] = useState<number>(3);

  // Processing state
  const [taskId, setTaskId] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [taskStatus, setTaskStatus] = useState<TaskStatusResponse | null>(null);
  const [highlightsData, setHighlightsData] = useState<TaskHighlightsResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const [isCheckingHealth, setIsCheckingHealth] = useState<boolean>(false);

  // Check backend health on mount
  useEffect(() => {
    void verifyHealth();
  }, []);

  const verifyHealth = async () => {
    setIsCheckingHealth(true);
    try {
      const res = await checkBackendHealth();
      setBackendOnline(res.available);
      if (res.available) {
        setErrorMessage(null);
      }
    } finally {
      setIsCheckingHealth(false);
    }
  };

  // Poll task status while processing
  useEffect(() => {
    if (!taskId || !isProcessing) return;

    const interval = setInterval(() => {
      void (async () => {
        try {
          const status = await getTaskStatus(taskId);
          setTaskStatus(status);

          if (status.status === "completed") {
            setIsProcessing(false);
            clearInterval(interval);
            toast.success("Highlight extraction and rendering complete!");
            // Fetch final highlights
            const hl = await getTaskHighlights(taskId);
            setHighlightsData(hl);
            onProcessingCompleteRef.current?.();
          } else if (status.status === "failed") {
            setIsProcessing(false);
            clearInterval(interval);
            setErrorMessage(status.error ?? "Processing failed");
            toast.error("Processing failed: " + (status.error ?? "Unknown error"));
          }
        } catch (err) {
          setIsProcessing(false);
          clearInterval(interval);
          const msg = err instanceof Error ? err.message : "Failed to poll task status";
          setErrorMessage(msg);
          toast.error(msg);
        }
      })();
    }, 800);

    return () => clearInterval(interval);
  }, [taskId, isProcessing]);

  // Handle video processing start
  const handleStartProcessing = async () => {
    setErrorMessage(null);
    setHighlightsData(null);
    setTaskStatus(null);

    if (!uploadedFile) {
      toast.error("Please select a video file to upload");
      return;
    }

    if (topK < 1 || topK > 10) {
      toast.error("Number of clips must be between 1 and 10");
      return;
    }

    setIsProcessing(true);

    try {
      toast.loading("Uploading video to ClipSense...", { id: "proc-toast" });
      const result = await submitVideoUpload(uploadedFile, {
        top_k: topK,
        use_cache: false,
        render_mode: "accurate",
        aspect_ratio: "9:16",
        add_subtitles: true,
        user_id: session?.user?.id,
      });

      toast.success("Processing queued: " + result.video_filename, { id: "proc-toast" });
      setTaskId(result.task_id);
    } catch (err) {
      setIsProcessing(false);
      const msg = err instanceof Error ? err.message : "Failed to start processing";
      setErrorMessage(msg);
      toast.error(msg, { id: "proc-toast" });
    }
  };

  // Stage display helper
  const getStageTitle = (stage?: string) => {
    switch (stage) {
      case "queued":
        return "1. Queued in Worker Pool";
      case "extraction":
        return "2. Multimodal Extraction (WhisperX, Frames, Prosody, Turns)";
      case "experts":
        return "3. Modality Experts (Transcript, Dialogue, Visual, Acoustics)";
      case "bundle":
        return "4. Common Evidence Bundle & Temporal Grid Assembly";
      case "mter":
        return "5. MTER Consensus Reasoning & Conflict Arbitration";
      case "rendering":
        return "6. FFmpeg Physical Video Slicing & Clip Export";
      case "completed":
        return "Highlights Exported & Ready for Playback";
      case "failed":
        return "Processing Error";
      default:
        return "Initializing Pipeline...";
    }
  };

  return (
    <div className="w-full max-w-6xl mx-auto space-y-8 py-6">
      {/* 2. VIDEO INPUT & CONFIGURATION CARD */}
      <Card className="border-slate-200 shadow-sm">
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Film className="h-5 w-5 text-blue-600" />
            Upload Video & Generate Clips
          </CardTitle>
          <CardDescription>
            Upload your raw video file (.mp4, .mov, .mkv, .webm up to 500 MB) to extract highlights.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* File Upload Dropzone */}
          <div
            className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors ${
              uploadedFile
                ? "border-green-400 bg-green-50/20"
                : "border-slate-300 hover:border-blue-400 bg-slate-50/50"
            }`}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="video/mp4,video/quicktime,video/x-matroska,video/webm"
              className="hidden"
              onChange={(e) => {
                if (e.target.files?.[0]) {
                  setUploadedFile(e.target.files[0]);
                }
              }}
            />
            <UploadCloud className="h-12 w-12 text-slate-400 mx-auto mb-3" />
            {uploadedFile ? (
              <div>
                <p className="text-base font-semibold text-slate-900">{uploadedFile.name}</p>
                <p className="text-xs text-slate-500 mt-1">
                  {(uploadedFile.size / (1024 * 1024)).toFixed(2)} MB — Click to choose a different file
                </p>
              </div>
            ) : (
              <div>
                <p className="text-base font-semibold text-slate-700">
                  Click to select or drag and drop a video
                </p>
                <p className="text-xs text-slate-500 mt-1">
                  MP4, MOV, MKV, AVI or WebM up to 500 MB
                </p>
              </div>
            )}
          </div>

          {/* Configuration: Number of Clips Only (min 1, max 10) */}
          <div className="pt-2">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-4 rounded-xl border bg-slate-50/70">
              <div className="space-y-0.5">
                <Label htmlFor="top-k" className="text-sm font-semibold flex items-center gap-2 text-slate-900">
                  <Sliders className="h-4 w-4 text-blue-600" /> Number of Clips (1 – 10)
                </Label>
                <p className="text-xs text-slate-500">
                  Specify how many highlight clips to generate (minimum 1, maximum 10).
                </p>
              </div>
              <div className="flex items-center gap-3">
                <Input
                  id="top-k"
                  type="number"
                  min={1}
                  max={10}
                  value={topK}
                  onChange={(e) => {
                    const parsed = parseInt(e.target.value);
                    if (isNaN(parsed)) {
                      setTopK(1);
                    } else {
                      setTopK(Math.max(1, Math.min(10, parsed)));
                    }
                  }}
                  className="h-10 w-24 text-center font-bold text-base bg-white"
                />
                <span className="text-xs text-slate-600 font-medium whitespace-nowrap">
                  {topK} {topK === 1 ? "clip" : "clips"}
                </span>
              </div>
            </div>
          </div>

          {/* Backend Connection Status Notice (Only shown if offline/unreachable) */}
          {backendOnline === false && (
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 p-3.5 rounded-xl border border-amber-200 bg-amber-50/70 text-amber-900 text-xs">
              <div className="flex items-start gap-2.5">
                <AlertCircle className="h-4 w-4 text-amber-600 mt-0.5 shrink-0" />
                <div>
                  <p className="font-semibold text-amber-900">ClipSense Engine Notice</p>
                  <p className="text-amber-800 mt-0.5">
                    Backend service is currently unreachable or waking up from idle. Please allow a moment and retry.
                  </p>
                </div>
              </div>
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={isCheckingHealth}
                onClick={() => void verifyHealth()}
                className="h-8 text-xs border-amber-300 bg-white hover:bg-amber-100/70 text-amber-900 shrink-0 font-medium"
              >
                {isCheckingHealth ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" />
                ) : (
                  <RefreshCw className="h-3.5 w-3.5 mr-1.5" />
                )}
                Retry Connection
              </Button>
            </div>
          )}

          {/* Action Button */}
          <div className="pt-2">
            <Button
              size="lg"
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold h-12 shadow-md shadow-blue-500/20"
              onClick={handleStartProcessing}
              disabled={isProcessing || !uploadedFile}
            >
              {isProcessing ? (
                <>
                  <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  Processing Video with ClipSense...
                </>
              ) : (
                <>
                  <Sparkles className="mr-2 h-5 w-5" />
                  Generate {topK} {topK === 1 ? "Clip" : "Clips"} with ClipSense
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* 3. REAL-TIME PROCESSING STATUS CARD */}
      {(isProcessing || taskStatus) && (
        <Card className="border-blue-200 bg-blue-50/20 shadow-sm">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-md flex items-center gap-2">
                {isProcessing ? (
                  <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
                ) : taskStatus?.status === "completed" ? (
                  <CheckCircle2 className="h-4 w-4 text-green-600" />
                ) : (
                  <AlertCircle className="h-4 w-4 text-red-600" />
                )}
                Pipeline Status: {taskStatus?.status?.toUpperCase() ?? "PROCESSING"}
              </CardTitle>
              {taskId && <span className="text-[11px] font-mono text-slate-500">ID: {taskId}</span>}
            </div>
            <CardDescription className="text-xs">
              {getStageTitle(taskStatus?.stage)}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Progress bar */}
            <div className="w-full bg-slate-200 rounded-full h-2.5 overflow-hidden">
              <div
                className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
                style={{ width: `${Math.round((taskStatus?.progress ?? 0.05) * 100)}%` }}
              />
            </div>

            <div className="flex items-center justify-between text-xs text-slate-600">
              <span className="font-medium">{taskStatus?.message ?? "Running pipeline..."}</span>
              <span className="font-mono">{Math.round((taskStatus?.progress ?? 0.05) * 100)}%</span>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Error state */}
      {errorMessage && (
        <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-900 text-sm flex items-start gap-3">
          <AlertCircle className="h-5 w-5 text-red-600 shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold">Processing Failed</p>
            <p className="text-xs text-red-700 mt-0.5">{errorMessage}</p>
          </div>
        </div>
      )}

      {/* 4. RESULTS DISPLAY: RANKED HIGHLIGHTS & VIDEO PLAYERS */}
      {highlightsData && (
        <div className="space-y-6 pt-4 border-t">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
            <div>
              <h3 className="text-xl font-bold text-slate-900">
                Extracted Highlight Clips ({highlightsData.total_candidates})
              </h3>
              <p className="text-xs text-slate-500">
                Source: {highlightsData.video_filename} (Duration: {highlightsData.total_duration}s)
                {highlightsData.execution_time_sec && ` • Elapsed: ${highlightsData.execution_time_sec.toFixed(1)}s`}
              </p>
            </div>
            <Badge variant="secondary" className="text-xs">
              MTER Consensus Complete
            </Badge>
          </div>

          {highlightsData.highlights.length === 0 ? (
            <div className="p-12 text-center rounded-xl border border-dashed text-slate-500 bg-slate-50">
              <Film className="h-8 w-8 mx-auto mb-2 text-slate-400" />
              <p className="text-sm font-semibold">No highlight candidates satisfied constraints</p>
              <p className="text-xs mt-1">Try increasing Top-K or using a video with more conversational activity.</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {highlightsData.highlights.map((item) => (
                <HighlightResultCard
                  key={item.candidate_id}
                  highlight={item}
                  taskId={highlightsData.task_id}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Individual Highlight Result Card with HTML5 player, evidence bars, and download button.
 */
function HighlightResultCard({
  highlight,
  taskId,
}: {
  highlight: HighlightItem;
  taskId: string;
}) {
  const streamUrl = highlight.clip?.stream_url
    ? getClipStreamUrl(taskId, highlight.clip.filename)
    : null;
  const downloadUrl = highlight.clip?.download_url
    ? getClipDownloadUrl(taskId, highlight.clip.filename)
    : null;

  const ev = highlight.modality_evidence;

  return (
    <Card className="border-slate-200 shadow-md hover:shadow-lg transition-shadow overflow-hidden flex flex-col justify-between">
      <div>
        {/* Top Header Badge */}
        <div className="p-4 bg-slate-50 border-b flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Badge className="bg-blue-600 text-white font-bold text-xs">
              Rank #{highlight.rank}
            </Badge>
            <span className="text-xs font-semibold text-slate-900 font-mono">
              [{highlight.start_time.toFixed(2)}s → {highlight.end_time.toFixed(2)}s]
            </span>
          </div>

          <Badge variant="outline" className="text-xs font-medium">
            <Clock className="h-3 w-3 mr-1 text-slate-500" />
            {highlight.duration.toFixed(1)}s
          </Badge>
        </div>

        {/* Video Player Section */}
        <div className="p-4 bg-black flex items-center justify-center aspect-video relative">
          {streamUrl ? (
            <video
              src={streamUrl}
              controls
              preload="metadata"
              className="w-full h-full object-contain rounded"
            />
          ) : (
            <div className="text-xs text-slate-400 flex flex-col items-center gap-2">
              <Film className="h-8 w-8 text-slate-600" />
              <span>Video clip not available</span>
            </div>
          )}
        </div>

        {/* Metadata & Scores */}
        <CardContent className="p-5 space-y-4">
          <div className="flex items-center justify-between text-xs">
            <div>
              <span className="text-slate-500">Evidence Strength:</span>{" "}
              <span className="font-bold text-blue-600 font-mono">
                {(highlight.confidence_estimate * 100).toFixed(1)}%
              </span>
            </div>
            <div>
              <span className="text-slate-500">Agreement IoU:</span>{" "}
              <span className="font-bold text-slate-700 font-mono">
                {highlight.temporal_agreement_iou.toFixed(2)}
              </span>
            </div>
          </div>

          {/* Modality Support Breakdown */}
          <div className="space-y-2 border-t pt-3">
            <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider block">
              Modality Support Evidence
            </span>

            <div className="space-y-1.5">
              {/* Transcript */}
              <ModalityBar
                icon={<FileText className="h-3 w-3 text-blue-600" />}
                label="Transcript Claim"
                value={ev.transcript_support}
              />
              {/* Conversation */}
              <ModalityBar
                icon={<Users className="h-3 w-3 text-emerald-600" />}
                label="Conversation Turn"
                value={ev.conversation_support}
              />
              {/* Visual */}
              <ModalityBar
                icon={<Eye className="h-3 w-3 text-purple-600" />}
                label="Visual Shift & Motion"
                value={ev.visual_support}
              />
              {/* Prosody */}
              <ModalityBar
                icon={<Volume2 className="h-3 w-3 text-amber-600" />}
                label="Acoustic Emphasis"
                value={ev.prosody_support}
              />
            </div>
          </div>

          {/* Temporal Boundary Refinement */}
          {highlight.boundary_refinement?.refined && (
            <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-2.5 text-xs text-emerald-900 space-y-1">
              <div className="flex items-center gap-1 font-semibold text-[11px] text-emerald-800">
                <Sparkles className="h-3 w-3 text-emerald-600" />
                <span>Evidence-Aware Boundary Refinement</span>
              </div>
              <div className="text-[11px] text-emerald-700">
                Original: [{highlight.boundary_refinement.original_start.toFixed(2)}s → {highlight.boundary_refinement.original_end.toFixed(2)}s]
                {" • "}
                Δstart: {highlight.boundary_refinement.start_delta_sec >= 0 ? `+${highlight.boundary_refinement.start_delta_sec.toFixed(2)}s` : `${highlight.boundary_refinement.start_delta_sec.toFixed(2)}s`}
                {" • "}
                Δend: {highlight.boundary_refinement.end_delta_sec >= 0 ? `+${highlight.boundary_refinement.end_delta_sec.toFixed(2)}s` : `${highlight.boundary_refinement.end_delta_sec.toFixed(2)}s`}
              </div>
              <p className="text-[10px] text-emerald-600 italic">
                {highlight.boundary_refinement.rationale}
              </p>
            </div>
          )}

          {/* Rationale / Decision Summary */}
          {highlight.decision_summary && (
            <div className="space-y-1 bg-slate-50 p-3 rounded-lg border text-xs text-slate-700">
              <span className="font-semibold block text-[11px] text-slate-900">
                Reasoning Decision Summary:
              </span>
              <p className="line-clamp-3 leading-relaxed text-slate-600 italic">
                &ldquo;{highlight.decision_summary}&rdquo;
              </p>
            </div>
          )}
        </CardContent>
      </div>

      {/* Footer Action: Download */}
      <div className="p-4 bg-slate-50 border-t flex items-center justify-between">
        <span className="text-[11px] text-slate-500 font-mono">
          {highlight.clip?.file_size_bytes
            ? `${(highlight.clip.file_size_bytes / (1024 * 1024)).toFixed(2)} MB • MP4`
            : "MP4 Video Clip"}
        </span>

        {downloadUrl && (
          <Button size="sm" variant="outline" asChild className="gap-1.5 text-xs">
            <a href={downloadUrl} download={highlight.clip?.filename}>
              <Download className="h-3.5 w-3.5" />
              Download Clip
            </a>
          </Button>
        )}
      </div>
    </Card>
  );
}

function ModalityBar({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: number;
}) {
  const pct = Math.round(Math.min(1.0, Math.max(0.0, value)) * 100);
  return (
    <div className="flex items-center gap-2 text-xs">
      <div className="w-4 flex items-center justify-center shrink-0">{icon}</div>
      <span className="w-36 truncate text-slate-600 text-[11px]">{label}</span>
      <div className="flex-1 bg-slate-100 rounded-full h-1.5 overflow-hidden">
        <div
          className="bg-slate-700 h-1.5 rounded-full transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="w-9 text-right font-mono text-[10px] text-slate-500">{pct}%</span>
    </div>
  );
}
