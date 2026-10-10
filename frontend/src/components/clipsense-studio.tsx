"use client";

import React, { useEffect, useState, useRef } from "react";
import {
  AlertCircle,
  CheckCircle2,
  Clock,
  Download,
  Film,
  Loader2,
  Play,
  RefreshCw,
  Sliders,
  Sparkles,
  UploadCloud,
  Video,
  Volume2,
  Users,
  Eye,
  FileText,
} from "lucide-react";
import { toast } from "sonner";
import {
  checkBackendHealth,
  getClipDownloadUrl,
  getClipStreamUrl,
  getTaskHighlights,
  getTaskStatus,
  submitVideoPath,
  submitVideoUpload,
  type HealthResponse,
  type HighlightItem,
  type TaskHighlightsResponse,
  type TaskStatusResponse,
} from "~/lib/clipsense-api";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Switch } from "./ui/switch";

const DEMO_PRESET_VIDEOS = [
  {
    name: "Representative 90s Dialogue (Benchmark)",
    path: "tests/sample_conversational_90s.mp4",
    duration: "90s",
    desc: "Parent-child discipline dialogue with pre-extracted multimodal features for fast demo.",
  },
  {
    name: "Full Podcast Video (46 mins)",
    path: "test.mp4",
    duration: "46m 38s",
    desc: "Full-length long-form research video for comprehensive multimodal highlight extraction.",
  },
  {
    name: "Peter Thiel - Zero to One (Short)",
    path: "Peter Thiel Going from Zero to One.mp4",
    duration: "2m 53s",
    desc: "Monologue discussion with distinct vocal emphasis and speech pauses.",
  },
];

export function ClipSenseStudio() {
  // Backend connection state
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [checkingHealth, setCheckingHealth] = useState(false);

  // Input selection state
  const [inputMode, setInputMode] = useState<"preset" | "upload">("preset");
  const [selectedPreset, setSelectedPreset] = useState<string>("tests/sample_conversational_90s.mp4");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Configuration options
  const [topK, setTopK] = useState<number>(3);
  const [useCache, setUseCache] = useState<boolean>(true);
  const [renderMode, setRenderMode] = useState<"accurate" | "fast">("accurate");

  // Processing state
  const [taskId, setTaskId] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [taskStatus, setTaskStatus] = useState<TaskStatusResponse | null>(null);
  const [highlightsData, setHighlightsData] = useState<TaskHighlightsResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Check backend health on mount
  useEffect(() => {
    void verifyHealth();
  }, []);

  const verifyHealth = async () => {
    setCheckingHealth(true);
    const res = await checkBackendHealth();
    setBackendOnline(res.available);
    if (res.available && res.data) {
      setHealth(res.data);
    } else {
      setHealth(null);
    }
    setCheckingHealth(false);
  };

  // Poll task status while processing
  useEffect(() => {
    if (!taskId || !isProcessing) return;

    const interval = setInterval(async () => {
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
        } else if (status.status === "failed") {
          setIsProcessing(false);
          clearInterval(interval);
          setErrorMessage(status.error || "Processing failed");
          toast.error("Processing failed: " + (status.error || "Unknown error"));
        }
      } catch (err) {
        setIsProcessing(false);
        clearInterval(interval);
        const msg = err instanceof Error ? err.message : "Failed to poll task status";
        setErrorMessage(msg);
        toast.error(msg);
      }
    }, 800);

    return () => clearInterval(interval);
  }, [taskId, isProcessing]);

  // Handle video processing start
  const handleStartProcessing = async () => {
    setErrorMessage(null);
    setHighlightsData(null);
    setTaskStatus(null);
    setIsProcessing(true);

    try {
      let result: { task_id: string; message: string; video_filename: string };

      if (inputMode === "upload") {
        if (!uploadedFile) {
          toast.error("Please select a video file to upload");
          setIsProcessing(false);
          return;
        }
        toast.loading("Uploading video to local ClipSense API...", { id: "proc-toast" });
        result = await submitVideoUpload(uploadedFile, {
          top_k: topK,
          use_cache: useCache,
          render_mode: renderMode,
        });
      } else {
        toast.loading("Queuing video processing with local pipeline...", { id: "proc-toast" });
        result = await submitVideoPath(selectedPreset, {
          top_k: topK,
          use_cache: useCache,
          render_mode: renderMode,
        });
      }

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
        return "6. FFmpeg Physical Video Slicing & Clip Export (W8)";
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
      {/* 1. HEADER & HEALTH STATUS BAR */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 p-6 rounded-2xl bg-gradient-to-r from-blue-900/10 via-indigo-900/10 to-purple-900/10 border border-slate-200">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
              Week 9 Research Bridge
            </Badge>
            <Badge variant="secondary" className="font-mono text-xs">
              MTER v1.0 + Week 8 Video Renderer
            </Badge>
          </div>
          <h2 className="text-2xl font-bold tracking-tight text-slate-900">
            ClipSense AI Interactive Studio
          </h2>
          <p className="text-sm text-slate-600">
            Extract evidence-backed highlights directly through the local FastAPI research service.
          </p>
        </div>

        {/* Backend health pill */}
        <div className="flex items-center gap-3 bg-white px-4 py-2.5 rounded-xl border shadow-sm">
          <div className="flex items-center gap-2">
            <span
              className={`h-3 w-3 rounded-full ${
                backendOnline === true
                  ? "bg-green-500 animate-pulse"
                  : backendOnline === false
                  ? "bg-red-500"
                  : "bg-amber-400"
              }`}
            />
            <span className="text-xs font-semibold">
              {backendOnline === true
                ? "Local API Ready"
                : backendOnline === false
                ? "API Offline (port 8000)"
                : "Checking backend..."}
            </span>
          </div>

          {health && (
            <Badge variant="secondary" className="text-[10px]">
              FFmpeg: {health.ffmpeg_available ? "✓ Active" : "✗ Missing"}
            </Badge>
          )}

          <Button
            size="sm"
            variant="ghost"
            className="h-7 w-7 p-0"
            onClick={() => void verifyHealth()}
            disabled={checkingHealth}
            title="Refresh API health status"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${checkingHealth ? "animate-spin" : ""}`} />
          </Button>
        </div>
      </div>

      {/* Backend offline warning banner */}
      {backendOnline === false && (
        <div className="p-4 rounded-xl bg-amber-50 border border-amber-200 text-amber-900 flex items-start gap-3">
          <AlertCircle className="h-5 w-5 text-amber-600 shrink-0 mt-0.5" />
          <div className="text-sm space-y-1">
            <p className="font-semibold">Local ClipSense API service not detected</p>
            <p className="text-xs text-amber-800">
              Start the FastAPI bridge by executing the following in your backend directory:
            </p>
            <pre className="p-2 bg-amber-100/60 rounded text-xs font-mono select-all">
              python -m uvicorn api.server:app --port 8000 --reload
            </pre>
          </div>
        </div>
      )}

      {/* 2. VIDEO INPUT & CONFIGURATION CARD */}
      <Card className="border-slate-200 shadow-sm">
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Film className="h-5 w-5 text-blue-600" />
            1. Select Video & Configure Pipeline
          </CardTitle>
          <CardDescription>
            Choose a research dataset video or upload your own raw video (.mp4, .mov, .mkv).
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Mode Switcher */}
          <div className="flex gap-2 p-1 bg-slate-100 rounded-lg max-w-md">
            <button
              type="button"
              className={`flex-1 py-1.5 text-xs font-medium rounded-md transition-all ${
                inputMode === "preset"
                  ? "bg-white text-slate-900 shadow-sm"
                  : "text-slate-600 hover:text-slate-900"
              }`}
              onClick={() => setInputMode("preset")}
            >
              Preset Research Videos
            </button>
            <button
              type="button"
              className={`flex-1 py-1.5 text-xs font-medium rounded-md transition-all ${
                inputMode === "upload"
                  ? "bg-white text-slate-900 shadow-sm"
                  : "text-slate-600 hover:text-slate-900"
              }`}
              onClick={() => setInputMode("upload")}
            >
              Upload Local File
            </button>
          </div>

          {/* Mode A: Preset Selector */}
          {inputMode === "preset" && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {DEMO_PRESET_VIDEOS.map((vid) => {
                const isSelected = selectedPreset === vid.path;
                return (
                  <div
                    key={vid.path}
                    className={`cursor-pointer rounded-xl border p-4 transition-all ${
                      isSelected
                        ? "border-blue-500 bg-blue-50/50 ring-1 ring-blue-500"
                        : "border-slate-200 hover:border-slate-300 bg-white"
                    }`}
                    onClick={() => setSelectedPreset(vid.path)}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-semibold text-xs text-slate-900">{vid.name}</span>
                      <Badge variant="outline" className="text-[10px]">
                        {vid.duration}
                      </Badge>
                    </div>
                    <p className="text-xs text-slate-500 line-clamp-2">{vid.desc}</p>
                    <div className="mt-3 text-[10px] text-blue-600 font-mono flex items-center gap-1">
                      <Video className="h-3 w-3" />
                      {vid.path}
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Mode B: File Upload Dropzone */}
          {inputMode === "upload" && (
            <div
              className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
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
                  if (e.target.files && e.target.files[0]) {
                    setUploadedFile(e.target.files[0]);
                  }
                }}
              />
              <UploadCloud className="h-10 w-10 text-slate-400 mx-auto mb-3" />
              {uploadedFile ? (
                <div>
                  <p className="text-sm font-semibold text-slate-900">{uploadedFile.name}</p>
                  <p className="text-xs text-slate-500 mt-1">
                    {(uploadedFile.size / (1024 * 1024)).toFixed(2)} MB — Click to choose a different file
                  </p>
                </div>
              ) : (
                <div>
                  <p className="text-sm font-semibold text-slate-700">
                    Click to select or drag and drop a video
                  </p>
                  <p className="text-xs text-slate-500 mt-1">
                    MP4, MOV, MKV, AVI or WebM up to 500 MB
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Configuration Controls */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-4 border-t">
            {/* Top-K Selector */}
            <div className="space-y-2">
              <Label htmlFor="top-k" className="text-xs font-semibold flex items-center gap-1.5">
                <Sliders className="h-3.5 w-3.5 text-blue-600" /> Top-K Highlights
              </Label>
              <div className="flex items-center gap-3">
                <Input
                  id="top-k"
                  type="number"
                  min={1}
                  max={5}
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value) || 1)}
                  className="h-9 w-20 text-center"
                />
                <span className="text-xs text-slate-500">Extracts top {topK} candidates</span>
              </div>
            </div>

            {/* Fast Cache Switch */}
            <div className="space-y-2">
              <Label className="text-xs font-semibold">Reuse Pre-computed Cache</Label>
              <div className="flex items-center justify-between h-9 px-3 border rounded-lg bg-slate-50">
                <span className="text-xs text-slate-600">Avoid re-extracting features</span>
                <Switch checked={useCache} onCheckedChange={setUseCache} />
              </div>
            </div>

            {/* Render Mode Switch */}
            <div className="space-y-2">
              <Label className="text-xs font-semibold">FFmpeg Render Mode</Label>
              <div className="flex items-center gap-2">
                <Button
                  type="button"
                  size="sm"
                  variant={renderMode === "accurate" ? "default" : "outline"}
                  className="h-9 flex-1 text-xs"
                  onClick={() => setRenderMode("accurate")}
                >
                  Accurate (H.264)
                </Button>
                <Button
                  type="button"
                  size="sm"
                  variant={renderMode === "fast" ? "default" : "outline"}
                  className="h-9 flex-1 text-xs"
                  onClick={() => setRenderMode("fast")}
                >
                  Fast Copy
                </Button>
              </div>
            </div>
          </div>

          {/* Action Button */}
          <div className="pt-2">
            <Button
              size="lg"
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold h-12 shadow-md shadow-blue-500/20"
              onClick={handleStartProcessing}
              disabled={isProcessing || backendOnline === false}
            >
              {isProcessing ? (
                <>
                  <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  Processing Video with ClipSense AI...
                </>
              ) : (
                <>
                  <Sparkles className="mr-2 h-5 w-5" />
                  Start Multimodal Extraction & Slicing
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
                Pipeline Status: {taskStatus?.status?.toUpperCase() || "PROCESSING"}
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
              <span className="font-medium">{taskStatus?.message || "Running pipeline..."}</span>
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
              <span className="text-slate-500">MTER Evidence Strength:</span>{" "}
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

          {/* Temporal Boundary Refinement (W10) */}
          {highlight.boundary_refinement?.refined && (
            <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-2.5 text-xs text-emerald-900 space-y-1">
              <div className="flex items-center gap-1 font-semibold text-[11px] text-emerald-800">
                <Sparkles className="h-3 w-3 text-emerald-600" />
                <span>Evidence-Aware Boundary Refinement (W10)</span>
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
