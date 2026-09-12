"use client";

import { Check, Copy, Download, Loader2, Send, Clock, X as CloseIcon, Youtube, Instagram, Facebook, Lock, Sparkles, UploadCloud, Plus, Trash2 } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import Dropzone, { type DropzoneState } from "shadcn-dropzone";
import { toast } from "sonner";
import { processVideo, getClipsPaginated } from "~/actions/generation";
import { generateUploadUrl } from "~/actions/s3";
import { ClipDisplay } from "./clip-display";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "./ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { Label } from "./ui/label";
import { Input } from "./ui/input";
import { Switch } from "./ui/switch";
import { Textarea } from "./ui/textarea";
import { signIn } from "next-auth/react";

type ClipWithUrl = {
    id: string;
    s3Key: string;
    title: string | null;
    description: string | null;
    createdAt: Date;
    playUrl: string;

    socialPosts?: {
        id: string;
        platform: string;
        status: string;
        scheduledFor: Date | null;
    }[];
};

export function DashboardClient({
    uploadedFiles,
    totalClipsCount,
    userCredits,
}: {
    uploadedFiles: {
        id: string;
        s3Key: string;
        filename: string;
        status: string;
        clipsCount: number;
        createdAt: Date;
    }[];
    totalClipsCount: number;
    userCredits: number;
}) {
    // --- UPLOAD & GENERATION STATE ---
    const [files, setFiles] = useState<File[]>([]);
    const [uploading, setUploading] = useState(false);
    const [refreshing, setRefreshing] = useState(false);
    const [isAutoMode, setIsAutoMode] = useState(true);
    const [requestedClips, setRequestedClips] = useState(3);
    const [customPrompt, setCustomPrompt] = useState("");
    const [videoDuration, setVideoDuration] = useState<number | null>(null);
    const [maxAllowedClips, setMaxAllowedClips] = useState<number | null>(null);

    // --- ZERO-CLICK PUBLISHING STATE ---
    const [autoPublish, setAutoPublish] = useState(false);
    const [publishMode, setPublishMode] = useState<"draft" | "direct">("draft");
    const [platforms, setPlatforms] = useState({
        youtube: true, instagram: true, facebook: false
        // , x: false
    });
    const [startDate, setStartDate] = useState(""); // YYYY-MM-DD
    const [timeSlots, setTimeSlots] = useState<string[]>(["10:00", "14:00"]); // For Auto Mode
    const [exactDates, setExactDates] = useState<string[]>([]); // For Manual Mode (Array of ISO strings)
    const [suffixes, setSuffixes] = useState({
        yt: "", insta: "", fb: ""
        // , x: "" 
    });
    const [manualScheduleType, setManualScheduleType] = useState<"pattern" | "exact">("pattern");

    // --- CLIPS DISPLAY STATE ---
    const [clips, setClips] = useState<ClipWithUrl[]>([]);
    const [currentPage, setCurrentPage] = useState(0);
    const [hasMore, setHasMore] = useState(totalClipsCount > 12);
    const [loadingClips, setLoadingClips] = useState(false);
    const router = useRouter();
    const [connections, setConnections] = useState({
        youtubeLinked: false, metaLinked: false,
        //  xLinked: false, 
        canConnectMeta: false, hasRequestedInvite: false, isLoading: true
    });
    const [activeSuffixTab, setActiveSuffixTab] = useState<"youtube" | "instagram" | "facebook" 
    // | "x"
    >("youtube");

    // Auto-switch suffix tabs if the user unchecks the currently active platform
    useEffect(() => {
        if (!platforms[activeSuffixTab]) {
            if (platforms.youtube) setActiveSuffixTab("youtube");
            else if (platforms.instagram) setActiveSuffixTab("instagram");
            else if (platforms.facebook) setActiveSuffixTab("facebook");
            // else if (platforms.x) setActiveSuffixTab("x");
        }
    }, [platforms, activeSuffixTab]);

    useEffect(() => {
        fetch("/api/social/accounts")
            // Add the explicit type definition right here:
            .then((res) => res.json() as Promise<{
                youtubeLinked: boolean;
                metaLinked: boolean;
                // xLinked: boolean;
                canConnectMeta?: boolean;
                hasRequestedInvite?: boolean;
            }>)
            .then((data) => {
                setConnections({
                    youtubeLinked: data.youtubeLinked,
                    metaLinked: data.metaLinked,
                    // xLinked: data.xLinked,
                    canConnectMeta: data.canConnectMeta ?? false,
                    hasRequestedInvite: data.hasRequestedInvite ?? false,
                    isLoading: false
                });

                // If in dashboard-client, you also have this:
                // setPlatforms({ youtube: data.youtubeLinked, instagram: data.metaLinked, facebook: data.metaLinked });
            })
            .catch(() => setConnections(prev => ({ ...prev, isLoading: false })));
    }, []);

    useEffect(() => {
        if (!isAutoMode) {
            setExactDates(Array(requestedClips).fill(""));
        }
    }, [isAutoMode, requestedClips]);

    useEffect(() => {
        if (totalClipsCount > 0) {
            void loadClips(0);
        }
    }, [totalClipsCount]);

    const loadClips = async (page: number) => {
        setLoadingClips(true);
        try {
            const result = await getClipsPaginated(page, 12);
            if (result.success && result.clips) {
                setClips(prev => page === 0 ? result.clips! : [...prev, ...result.clips!]);
                setHasMore(result.hasMore);
                setCurrentPage(page);
            }
        } catch (error) {
            toast.error("Failed to load clips");
        } finally {
            setLoadingClips(false);
        }
    };

    const handleLoadMore = () => { void loadClips(currentPage + 1); };

    const handleRefresh = async () => {
        setRefreshing(true);
        router.refresh();
        setTimeout(() => setRefreshing(false), 600);
    };

    const handleDrop = (acceptedFiles: File[]) => {
        setFiles(acceptedFiles);
        if (acceptedFiles.length > 0) {
            const file = acceptedFiles[0]!;
            const video = document.createElement('video');
            video.preload = 'metadata';
            video.onloadedmetadata = () => {
                const duration = video.duration;
                setVideoDuration(duration);
                const densityLimit = Math.max(1, Math.floor(duration / 120));
                const hardCap = 10;
                const maxClips = Math.min(densityLimit, hardCap, userCredits);
                setMaxAllowedClips(maxClips);
                setRequestedClips(Math.min(requestedClips, maxClips));
                URL.revokeObjectURL(video.src);
            };
            video.src = URL.createObjectURL(file);
        }
    };

    // --- AUTO-PUBLISH HANDLERS ---
    const handleAddTimeSlot = () => {
        if (timeSlots.length >= 5) return;
        setTimeSlots([...timeSlots, "12:00"]);
    };

    const handleRemoveTimeSlot = (index: number) => {
        setTimeSlots(timeSlots.filter((_, i) => i !== index));
    };

    const handleTimeChange = (index: number, newTime: string) => {
        const updated = [...timeSlots];
        updated[index] = newTime;
        setTimeSlots(updated);
    };

    const handleUpload = async () => {
        if (files.length === 0) return;
        const file = files[0]!;

        // Auto-Publish Validation
        if (autoPublish) {
            if (!platforms.youtube && !platforms.instagram && !platforms.facebook) {
                toast.error("Please select at least one publishing platform.");
                return;
            }

            if (!isAutoMode && manualScheduleType === "exact") {
                // EXACT MODE: Check if ANY of the exact dates are left blank
                if (exactDates.some(date => !date)) {
                    toast.error("Please fill out the exact date and time for every clip.");
                    return;
                }
            } else {
                // PATTERN MODE: Check if time slots and start date exist
                if (timeSlots.length === 0) {
                    toast.error("Please select at least one daily time slot.");
                    return;
                }
                if (!startDate) {
                    toast.error("Please select a Start Date.");
                    return;
                }
            }
        }

        setUploading(true);

        try {
            const { success, signedUrl, uploadedFileId } = await generateUploadUrl({ filename: file.name, contentType: file.type });
            if (!success) throw new Error("Failed to get upload URL");

            const uploadResponse = await fetch(signedUrl, {
                method: "PUT",
                body: file,
                headers: { "Content-Type": file.type },
            });

            if (!uploadResponse.ok) throw new Error(`Upload failed with status: ${uploadResponse.status}`);

            // Send generation configuration to the backend
            await processVideo(uploadedFileId, {
                mode: isAutoMode ? "auto" : "manual",
                requestedClips: isAutoMode ? undefined : requestedClips,
                userPrompt: customPrompt.trim() !== "" ? customPrompt : undefined,

                autoPublish,
                publishMode: autoPublish ? publishMode : undefined,
                publishPlatforms: autoPublish ? platforms : undefined,

                // Pass 'exactDates' if they chose exact, otherwise pass 'timeSlots'
                publishTimeSlots: autoPublish ? (!isAutoMode && manualScheduleType === "exact" ? exactDates : timeSlots) : undefined,

                // Tell the backend which method they used
                scheduleType: autoPublish ? (isAutoMode ? "pattern" : manualScheduleType) : undefined,
                startDate: autoPublish && (isAutoMode || manualScheduleType === "pattern") ? startDate : undefined,

                // Separated Suffixes
                ytSuffix: autoPublish ? suffixes.yt : undefined,
                instaSuffix: autoPublish ? suffixes.insta : undefined,
                fbSuffix: autoPublish ? suffixes.fb : undefined,
                // xSuffix: autoPublish ? suffixes.x : undefined,
            });

            setFiles([]);
            setCustomPrompt("");
            setAutoPublish(false); // Reset UI

            toast.success("Video uploaded successfully", {
                description: "Your video has been scheduled for processing. Check the status below.",
                duration: 5000,
            });
        } catch (error) {
            toast.error("Upload failed", {
                description: error instanceof Error ? error.message : "There was a problem uploading your video. Please try again.",
            });
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="mx-auto flex max-w-5xl flex-col space-y-6 px-4 py-8">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-semibold tracking-tight">Podcast Clipper</h1>
                    <p className="text-muted-foreground">Upload your podcast and get AI-generated clips instantly</p>
                </div>
                <div className="flex items-center gap-4">
                    <Link href="/dashboard/billing">
                        <Button>Buy Credits</Button>
                    </Link>
                </div>
            </div>

            <Tabs defaultValue="upload">
                <TabsList>
                    <TabsTrigger value="upload">Upload</TabsTrigger>
                    <TabsTrigger value="my-clips">My Clips</TabsTrigger>
                </TabsList>

                <TabsContent value="upload">
                    <Card>
                        <CardHeader>
                            <CardTitle>Upload Podcast</CardTitle>
                            <CardDescription>Upload your audio or video files to generate clips</CardDescription>
                            <div className="mt-2 rounded-lg bg-amber-50 border border-amber-200 p-3">
                                <p className="text-xs text-amber-800">
                                    ⚠️ <strong>AI Disclaimer:</strong> Clips, captions, and transcriptions are AI-generated and may contain errors. Always review content before publishing.
                                </p>
                            </div>
                        </CardHeader>
                        <CardContent>
                            <Dropzone
                                onDrop={handleDrop}
                                accept={{ "video/mp4": [".mp4"] }}
                                maxSize={500 * 1024 * 1024}
                                disabled={uploading}
                                maxFiles={1}
                            >
                                {(dropzone: DropzoneState) => (
                                    <div className="flex flex-col items-center justify-center space-y-4 rounded-lg p-10 text-center">
                                        <UploadCloud className="text-muted-foreground h-12 w-12" />
                                        <p className="font-medium">Drag and drop your file</p>
                                        <p className="text-muted-foreground text-sm">or click to browse (MP4 up to 500MB)</p>
                                        <Button className="cursor-pointer" variant="default" size="sm" disabled={uploading}>Select File</Button>
                                    </div>
                                )}
                            </Dropzone>

                            <div className="text-sm mt-2">
                                {files.length > 0 && (
                                    <span className="font-medium text-muted-foreground">Selected File: <span className="text-foreground">{files[0]?.name}</span></span>
                                )}
                            </div>

                            <div className="mt-6 space-y-8">
                                {/* --- CLIP GENERATION SETTINGS --- */}
                                <div className="space-y-4">
                                    <h3 className="text-sm font-bold uppercase tracking-wider text-muted-foreground">1. Generation Settings</h3>

                                    <div className="flex items-center justify-between">
                                        <div>
                                            <Label className="text-sm font-medium">Clip Mode</Label>
                                            <p className="text-xs text-muted-foreground">
                                                {isAutoMode ? "AI picks the best moments within your credit limit." : "You choose exactly how many clips to generate."}
                                            </p>
                                        </div>
                                        <div className="flex items-center rounded-lg border p-1 gap-1">
                                            <button type="button" onClick={() => setIsAutoMode(true)} className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${isAutoMode ? "bg-primary text-primary-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}>Auto</button>
                                            <button type="button" onClick={() => setIsAutoMode(false)} className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${!isAutoMode ? "bg-primary text-primary-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}>Manual</button>
                                        </div>
                                    </div>

                                    {!isAutoMode && (
                                        <div className="space-y-1">
                                            <Label className="text-sm font-medium">Number of clips</Label>
                                            <div className="flex items-center gap-3">
                                                <Input type="number" min={1} max={maxAllowedClips ?? 10} value={requestedClips} onChange={(e) => setRequestedClips(Math.max(1, parseInt(e.target.value) || 1))} className="w-20" disabled={uploading} />
                                                {maxAllowedClips && (
                                                    <p className="text-xs text-muted-foreground">Max {maxAllowedClips} clips {videoDuration && ` · ${Math.floor(videoDuration / 60)}m${Math.floor(videoDuration % 60)}s video`}</p>
                                                )}
                                            </div>
                                        </div>
                                    )}

                                    <div className="space-y-1">
                                        <Label className="text-sm font-medium">Topic Focus Prompt <span className="text-muted-foreground font-normal">(optional)</span></Label>
                                        <p className="text-xs text-muted-foreground">Tell the AI what to look for. If nothing matches, you won&apos;t be charged.</p>
                                        <Input placeholder="e.g. Find clips where he talks about marketing, failure, or AI..." value={customPrompt} onChange={(e) => setCustomPrompt(e.target.value)} className="w-full" disabled={uploading} />
                                    </div>
                                </div>

                                {/* --- ZERO-CLICK PUBLISHING SETTINGS --- */}
                                <div className="space-y-4 pt-4 border-t">
                                    <h3 className="text-sm font-bold uppercase tracking-wider text-muted-foreground">2. Scheduling Settings</h3>

                                    <div className="flex items-center justify-between bg-slate-50 border p-4 rounded-lg">
                                        <div>
                                            <Label className="text-base font-bold flex items-center gap-2">
                                                <Sparkles className="w-4 h-4 text-purple-500" /> Auto-Publish Clips
                                            </Label>
                                            <p className="text-sm text-muted-foreground mt-1">Automatically schedule generated clips to your socials.</p>
                                        </div>
                                        <Switch checked={autoPublish} onCheckedChange={setAutoPublish} disabled={uploading} />
                                    </div>

                                    {autoPublish && (
                                        <div className="space-y-6 bg-slate-50/50 p-4 border rounded-lg animate-in fade-in slide-in-from-top-4 duration-300">

                                            <div className="space-y-3">
                                                <Label className="font-semibold">Publishing Mode</Label>
                                                <div className="grid grid-cols-2 gap-3">
                                                    <button onClick={() => setPublishMode("draft")} className={`p-3 rounded-lg border text-left transition-all ${publishMode === "draft" ? "border-blue-500 bg-blue-50 ring-1 ring-blue-500" : "bg-white hover:bg-slate-50"}`}>
                                                        <div className="font-semibold text-sm text-blue-900">Save as Drafts</div>
                                                        <div className="text-xs text-blue-700/70 mt-1">We&apos;ll schedule them, but you must click &quot;Approve&quot; before they post.</div>
                                                    </button>
                                                    <button onClick={() => setPublishMode("direct")} className={`p-3 rounded-lg border text-left transition-all ${publishMode === "direct" ? "border-green-500 bg-green-50 ring-1 ring-green-500" : "bg-white hover:bg-slate-50"}`}>
                                                        <div className="font-semibold text-sm text-green-900">Direct Push</div>
                                                        <div className="text-xs text-green-700/70 mt-1">Post directly to your schedule without manual review.</div>
                                                    </button>
                                                </div>
                                            </div>

                                            <div className="space-y-3">
                                                <Label className="font-semibold">Platforms</Label>
                                                <div className="flex gap-4">
                                                    {/* Render YouTube */}
                                                    {connections.youtubeLinked ? (
                                                        <label className="flex items-center gap-2 text-sm font-medium cursor-pointer">
                                                            <input type="checkbox" className="rounded" checked={platforms.youtube} onChange={(e) => setPlatforms({ ...platforms, youtube: e.target.checked })} />
                                                            <Youtube className="w-4 h-4 text-red-500" /> YouTube
                                                        </label>
                                                    ) : (
                                                        <Button size="sm" variant="outline" onClick={() => signIn("google")}>Link YouTube</Button>
                                                    )}

                                                    {/* Render Meta (Insta/FB) */}
                                                    {connections.metaLinked ? (
                                                        <>
                                                            <label className="flex items-center gap-2 text-sm font-medium cursor-pointer">
                                                                <input type="checkbox" className="rounded" checked={platforms.instagram} onChange={(e) => setPlatforms({ ...platforms, instagram: e.target.checked })} />
                                                                <Instagram className="w-4 h-4 text-pink-500" /> Instagram
                                                            </label>
                                                            <label className="flex items-center gap-2 text-sm font-medium cursor-pointer">
                                                                <input type="checkbox" className="rounded" checked={platforms.facebook} onChange={(e) => setPlatforms({ ...platforms, facebook: e.target.checked })} />
                                                                <Facebook className="w-4 h-4 text-blue-600" /> Facebook
                                                            </label>
                                                        </>
                                                    ) : connections.canConnectMeta ? (
                                                        <Button size="sm" variant="outline" onClick={() => signIn("facebook")}>Link Meta</Button>
                                                    ) : (
                                                        <Button size="sm" variant="outline" disabled>Meta Invite Pending</Button>
                                                    )}
                                                    {/* {connections.xLinked ? (
                                                        <label className="flex items-center gap-2 text-sm font-medium cursor-pointer">
                                                            <input type="checkbox" className="rounded" checked={platforms.x} onChange={(e) => setPlatforms({ ...platforms, x: e.target.checked })} />
                                                            <img src='/x.svg' className="w-4 h-4 mr-2 text-slate-700" /> X
                                                        </label>
                                                    ) : (
                                                        <Button size="sm" variant="outline" onClick={() => signIn("twitter")} className="flex-1 bg-white hover:bg-slate-100 hover:text-slate-900 hover:border-slate-300">
                                                            <img src='/x.svg' className="w-4 h-4 mr-2 text-slate-700" /> Link X
                                                        </Button>
                                                    )} */}
                                                </div>
                                            </div>
                                            <div className="space-y-4">
                                                {/* If they are in Manual Mode, ask them HOW they want to schedule */}
                                                {!isAutoMode && (
                                                    <div className="space-y-3 pb-4 border-b border-slate-200">
                                                        <Label className="font-semibold">Scheduling Method</Label>
                                                        <div className="grid grid-cols-2 gap-3">
                                                            <button onClick={() => setManualScheduleType("pattern")} className={`p-3 rounded-lg border text-left transition-all ${manualScheduleType === "pattern" ? "border-primary bg-primary/5 ring-1 ring-primary" : "bg-white hover:bg-slate-50"}`}>
                                                                <div className="font-semibold text-sm">Daily Pattern</div>
                                                                <div className="text-xs text-muted-foreground mt-1">Auto-distribute across days.</div>
                                                            </button>
                                                            <button onClick={() => setManualScheduleType("exact")} className={`p-3 rounded-lg border text-left transition-all ${manualScheduleType === "exact" ? "border-primary bg-primary/5 ring-1 ring-primary" : "bg-white hover:bg-slate-50"}`}>
                                                                <div className="font-semibold text-sm">Exact Times</div>
                                                                <div className="text-xs text-muted-foreground mt-1">Pick a specific date for each clip.</div>
                                                            </button>
                                                        </div>
                                                    </div>
                                                )}

                                                {/* SHOW PATTERN UI (If Auto Mode OR Manual Mode + Pattern Selected) */}
                                                {isAutoMode || manualScheduleType === "pattern" ? (
                                                    <div className="space-y-4">
                                                        <div>
                                                            <Label className="font-semibold">Scheduling Pattern</Label>
                                                            <p className="text-xs text-muted-foreground mt-1">Clips will post sequentially on these times, starting on your selected date.</p>
                                                        </div>
                                                        <div className="flex items-center gap-4 mt-1">
                                                            <div>
                                                                <Label className="text-xs">Start Date</Label>
                                                                <Input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} className="bg-white" />
                                                            </div>
                                                        </div>
                                                        <div className="pt-2">
                                                            <div className="flex items-center justify-between">
                                                                <Label className="font-semibold">Daily Time Slots</Label>
                                                                <Button variant="outline" size="sm" onClick={handleAddTimeSlot} disabled={timeSlots.length >= 5} className="h-8 text-xs bg-white">
                                                                    <Plus className="w-3 h-3 mr-1" /> Add Slot
                                                                </Button>
                                                            </div>
                                                            <div className="space-y-2 mt-2">
                                                                {timeSlots.map((time, index) => (
                                                                    <div key={index} className="flex items-center gap-3">
                                                                        <Clock className="w-4 h-4 text-slate-400" />
                                                                        <Input type="time" value={time} onChange={(e) => handleTimeChange(index, e.target.value)} className="w-32 bg-white" />
                                                                        {timeSlots.length > 1 && (
                                                                            <Button variant="ghost" size="icon" onClick={() => handleRemoveTimeSlot(index)} className="text-red-400 hover:text-red-600 hover:bg-red-50">
                                                                                <Trash2 className="w-4 h-4" />
                                                                            </Button>
                                                                        )}
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        </div>
                                                    </div>
                                                ) : (
                                                    // SHOW EXACT TIMES UI (If Manual Mode + Exact Times Selected)
                                                    <div className="space-y-3">
                                                        <Label className="font-semibold">Schedule Your {requestedClips} Clips</Label>
                                                        <p className="text-xs text-muted-foreground">Set a precise date and time for each generated clip.</p>
                                                        <div className="space-y-2 mt-2">
                                                            {exactDates.map((date, index) => (
                                                                <div key={index} className="flex items-center gap-3">
                                                                    <Label className="w-16 text-xs text-muted-foreground">Clip {index + 1}</Label>
                                                                    <Input
                                                                        type="datetime-local"
                                                                        value={date}
                                                                        onChange={(e) => {
                                                                            const updated = [...exactDates];
                                                                            updated[index] = e.target.value;
                                                                            setExactDates(updated);
                                                                        }}
                                                                        className="bg-white"
                                                                    />
                                                                </div>
                                                            ))}
                                                        </div>
                                                    </div>
                                                )}
                                            </div>

                                            <div className="space-y-3">
                                                <Label className="font-semibold">Platform Suffixes & Hashtags</Label>
                                                <p className="text-xs text-muted-foreground -mt-1">We&apos;ll append this to the bottom of the AI-generated descriptions.</p>

                                                <div className="border rounded-lg overflow-hidden bg-white">
                                                    {/* TAB HEADERS */}
                                                    <div className="flex border-b bg-slate-50/50">
                                                        {platforms.youtube && connections.youtubeLinked && (
                                                            <button onClick={() => setActiveSuffixTab("youtube")} className={`flex-1 flex items-center justify-center gap-2 py-2 text-xs font-medium border-b-2 transition-colors ${activeSuffixTab === "youtube" ? "border-red-500 text-foreground bg-white" : "border-transparent text-muted-foreground hover:bg-slate-100"}`}>
                                                                <Youtube className="w-3 h-3 text-red-500" /> YouTube
                                                            </button>
                                                        )}
                                                        {platforms.instagram && connections.metaLinked && (
                                                            <button onClick={() => setActiveSuffixTab("instagram")} className={`flex-1 flex items-center justify-center gap-2 py-2 text-xs font-medium border-b-2 transition-colors ${activeSuffixTab === "instagram" ? "border-pink-500 text-foreground bg-white" : "border-transparent text-muted-foreground hover:bg-slate-100"}`}>
                                                                <Instagram className="w-3 h-3 text-pink-500" /> Instagram
                                                            </button>
                                                        )}
                                                        {platforms.facebook && connections.metaLinked && (
                                                            <button onClick={() => setActiveSuffixTab("facebook")} className={`flex-1 flex items-center justify-center gap-2 py-2 text-xs font-medium border-b-2 transition-colors ${activeSuffixTab === "facebook" ? "border-blue-600 text-foreground bg-white" : "border-transparent text-muted-foreground hover:bg-slate-100"}`}>
                                                                <Facebook className="w-3 h-3 text-blue-600" /> Facebook
                                                            </button>
                                                        )}
                                                        {/* {platforms.x && connections.xLinked && (
                                                            <button onClick={() => setActiveSuffixTab("x")} className={`flex-1 flex items-center justify-center gap-2 py-2 text-xs font-medium border-b-2 transition-colors ${activeSuffixTab === "x" ? "border-black-600 text-foreground bg-white" : "border-transparent text-muted-foreground hover:bg-slate-100"}`}>
                                                                <img src="/x.svg" className="w-3 h-3 text-blue-600" /> X
                                                            </button>
                                                        )} */}
                                                    </div>

                                                    {/* TAB CONTENT */}
                                                    <div className="p-2 pt-1">
                                                        {activeSuffixTab === "youtube" && platforms.youtube && (
                                                            <div className="animate-in fade-in zoom-in-95 duration-200">
                                                                <Textarea rows={2} value={suffixes.yt} onChange={(e) => setSuffixes({ ...suffixes, yt: e.target.value })} placeholder="Subscribe for more! #podcast #shorts" className="border-0 focus-visible:ring-0 px-1 resize-none" />
                                                            </div>
                                                        )}
                                                        {activeSuffixTab === "instagram" && platforms.instagram && (
                                                            <div className="animate-in fade-in zoom-in-95 duration-200">
                                                                <Textarea rows={2} value={suffixes.insta} onChange={(e) => setSuffixes({ ...suffixes, insta: e.target.value })} placeholder="Link in bio! #reels #business" className="border-0 focus-visible:ring-0 px-1 resize-none" />
                                                            </div>
                                                        )}
                                                        {activeSuffixTab === "facebook" && platforms.facebook && (
                                                            <div className="animate-in fade-in zoom-in-95 duration-200">
                                                                <Textarea rows={2} value={suffixes.fb} onChange={(e) => setSuffixes({ ...suffixes, fb: e.target.value })} placeholder="Follow our page for daily clips!" className="border-0 focus-visible:ring-0 px-1 resize-none" />
                                                            </div>
                                                        )}
                                                        {/* {activeSuffixTab === "x" && platforms.x && (
                                                            <div className="animate-in fade-in zoom-in-95 duration-200">
                                                                <Textarea rows={2} value={suffixes.x} onChange={(e) => setSuffixes({ ...suffixes, x: e.target.value })} placeholder="Follow our account for daily clips!" className="border-0 focus-visible:ring-0 px-1 resize-none" />
                                                            </div>
                                                        )} */}
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </div>

                                {/* --- ERRORS AND SUBMIT BUTTON --- */}
                                <div className="pt-1">
                                    {userCredits === 0 && <p className="text-sm text-destructive mb-3">No credits available. Buy credits to generate clips.</p>}
                                    {!isAutoMode && requestedClips > userCredits && userCredits > 0 && <p className="text-sm text-destructive mb-3">You have {userCredits} credits but requested {requestedClips} clips.</p>}

                                    <div className="flex items-center justify-end">
                                        <Button
                                            size="lg"
                                            disabled={files.length === 0 || uploading || (userCredits === 0) || (!isAutoMode && requestedClips > userCredits)}
                                            onClick={handleUpload}
                                        >
                                            {uploading ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Uploading...</> : "Upload and Generate Clips"}
                                        </Button>
                                    </div>
                                </div>
                            </div>

                            {/* --- QUEUE STATUS TABLE --- */}
                            {uploadedFiles.length > 0 && (
                                <div className="pt-10 mt-6 border-t">
                                    <div className="mb-4 flex items-center justify-between">
                                        <h3 className="text-lg font-semibold">Queue Status</h3>
                                        <Button variant="outline" size="sm" onClick={handleRefresh}>
                                            {refreshing ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Refreshing...</> : "Refresh"}
                                        </Button>
                                    </div>
                                    <div className="max-h-75 overflow-auto rounded-md border">
                                        <Table>
                                            <TableHeader>
                                                <TableRow>
                                                    <TableHead>File</TableHead>
                                                    <TableHead>Uploaded</TableHead>
                                                    <TableHead>Status</TableHead>
                                                    <TableHead>Clips created</TableHead>
                                                </TableRow>
                                            </TableHeader>
                                            <TableBody>
                                                {uploadedFiles.map((item) => (
                                                    <TableRow key={item.id}>
                                                        <TableCell className="max-w-xs truncate font-medium">{item.filename}</TableCell>
                                                        <TableCell className="text-muted-foreground text-sm">
                                                            {new Date(item.createdAt).toLocaleDateString("en-US", { year: "numeric", month: "numeric", day: "numeric" })}
                                                        </TableCell>
                                                        <TableCell>
                                                            {item.status === "queued" && <Badge variant="outline">Queued</Badge>}
                                                            {item.status === "processing" && <Badge variant="outline">Processing</Badge>}
                                                            {item.status === "processed" && <Badge variant="outline">Processed</Badge>}
                                                            {item.status === "no credits" && <Badge variant="destructive">No credits</Badge>}
                                                            {item.status === "failed" && <Badge variant="destructive">Failed</Badge>}
                                                            {item.status === "no_match" && <Badge variant="secondary" className="bg-yellow-500">No Match Found</Badge>}
                                                        </TableCell>
                                                        <TableCell>
                                                            {item.clipsCount > 0 ? (
                                                                <span>{item.clipsCount} clip{item.clipsCount !== 1 ? "s" : ""}</span>
                                                            ) : item.status === "no_match" ? (
                                                                <span className="text-muted-foreground">Topic not found.</span>
                                                            ) : (<span className="text-muted-foreground">No clips yet</span>)}
                                                        </TableCell>
                                                    </TableRow>
                                                ))}
                                            </TableBody>
                                        </Table>
                                    </div>
                                </div>
                            )}
                        </CardContent>
                    </Card>
                </TabsContent>

                <TabsContent value="my-clips">
                    <Card>
                        <CardHeader>
                            <CardTitle>My Clips</CardTitle>
                            <CardDescription>View and manage your generated clips here. Processing may take a few minutes.</CardDescription>
                            <div className="mt-2 rounded-lg bg-blue-50 border border-blue-200 p-3">
                                <p className="text-xs text-blue-800">
                                    🤖 <strong>AI Notice:</strong> All clips are automatically generated. Verify accuracy of captions and content before sharing publicly.
                                </p>
                            </div>
                        </CardHeader>
                        <CardContent>
                            <ClipDisplay clips={clips} onLoadMore={handleLoadMore} hasMore={hasMore} isLoading={loadingClips} />
                        </CardContent>
                    </Card>
                </TabsContent>
            </Tabs>
        </div>
    );
}