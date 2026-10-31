"use client";

import { Download, Loader2, Sparkles, RefreshCw } from "lucide-react";
import Link from "next/link";
import { useEffect, useState, useCallback } from "react";
import { ClipSenseStudio } from "./clipsense-studio";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { useSession } from "next-auth/react";
import { getUserClips, type UserClipItem } from "~/lib/clipsense-api";

function getClipRank(clip: UserClipItem, fallbackIndex: number): number {
    if (typeof clip.rank === "number" && clip.rank > 0) return clip.rank;
    const match = /highlight_0*(\d+)/i.exec(clip.filename);
    if (match?.[1]) {
        const parsed = parseInt(match[1], 10);
        if (!isNaN(parsed) && parsed > 0) return parsed;
    }
    return fallbackIndex + 1;
}

export function DashboardClient({
    userCredits: _userCredits = 0,
    userId: propUserId,
}: {
    userCredits?: number;
    userId?: string;
    uploadedFiles?: unknown[];
    totalClipsCount?: number;
}) {
    const { data: session } = useSession();
    const userId = propUserId ?? session?.user?.id ?? (session?.user?.email ? encodeURIComponent(session.user.email) : "default_user");

    const [activeTab, setActiveTab] = useState<string>("clipsense");

    // ClipSense local clips state
    const [localUserClips, setLocalUserClips] = useState<UserClipItem[]>([]);
    const [loadingLocalClips, setLoadingLocalClips] = useState(false);

    const loadUserLocalClips = useCallback(async () => {
        if (!userId) return;
        setLoadingLocalClips(true);
        try {
            const data = await getUserClips(userId);
            const clips = (data.clips ?? []).map((clip, idx) => ({
                ...clip,
                rank: getClipRank(clip, idx),
            }));
            // Sort: newest batches first, but within the same task sort ascending by rank (1, 2, 3)
            clips.sort((a, b) => {
                if (a.task_id && b.task_id && a.task_id === b.task_id) {
                    return (a.rank ?? 0) - (b.rank ?? 0);
                }
                return (b.created_at ?? 0) - (a.created_at ?? 0);
            });
            setLocalUserClips(clips);
        } catch {
            // Silently handle if backend is not yet started or has no clips
            setLocalUserClips([]);
        } finally {
            setLoadingLocalClips(false);
        }
    }, [userId]);

    useEffect(() => {
        void loadUserLocalClips();
    }, [loadUserLocalClips]);

    return (
        <div className="mx-auto flex max-w-5xl flex-col space-y-6 px-4 py-8">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-semibold tracking-tight text-slate-900">ClipSense Studio</h1>
                    <p className="text-muted-foreground text-sm">
                        Extract multimodal highlights, active speaker vertical crops, and captions with ClipSense
                    </p>
                </div>
                <div className="flex items-center gap-4">
                    <Link href="/dashboard/billing">
                        <Button>Buy Credits</Button>
                    </Link>
                </div>
            </div>

            <Tabs
                value={activeTab}
                onValueChange={(val) => {
                    setActiveTab(val);
                    if (val === "my-clips") {
                        void loadUserLocalClips();
                    }
                }}
            >
                <TabsList className="bg-slate-100 p-1 rounded-lg">
                    <TabsTrigger
                        value="clipsense"
                        className="flex items-center gap-1.5 font-semibold text-blue-600 data-[state=active]:bg-white data-[state=active]:text-blue-700"
                    >
                        <Sparkles className="h-3.5 w-3.5" /> ClipSense Studio
                    </TabsTrigger>
                    <TabsTrigger value="my-clips">My Clips</TabsTrigger>
                </TabsList>

                {/* Tab 1: ClipSense Studio - forceMount preserves state across tab changes */}
                <TabsContent value="clipsense" forceMount className="data-[state=inactive]:hidden">
                    <ClipSenseStudio onProcessingComplete={() => void loadUserLocalClips()} />
                </TabsContent>

                {/* Tab 2: My Clips (ClipSense generated clips only) */}
                <TabsContent value="my-clips" forceMount className="data-[state=inactive]:hidden">
                    <Card className="border-slate-200 shadow-sm">
                        <CardHeader>
                            <div className="flex items-center justify-between">
                                <div>
                                    <CardTitle className="text-lg">My Clips</CardTitle>
                                    <CardDescription>
                                        Highlights extracted and rendered by ClipSense
                                    </CardDescription>
                                </div>
                                <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={loadUserLocalClips}
                                    disabled={loadingLocalClips}
                                >
                                    <RefreshCw className={`h-3.5 w-3.5 mr-1.5 ${loadingLocalClips ? "animate-spin" : ""}`} />
                                    Refresh
                                </Button>
                            </div>
                        </CardHeader>
                        <CardContent className="space-y-6">
                            <div className="space-y-4">
                                <div className="flex items-center justify-between border-b pb-2">
                                    <div>
                                        <h3 className="text-base font-semibold flex items-center gap-2">
                                            <Sparkles className="h-4 w-4 text-blue-600" />
                                            ClipSense Saved Clips
                                            {localUserClips.length > 0 && (
                                                <Badge variant="secondary" className="text-xs">
                                                    {localUserClips.length} {localUserClips.length === 1 ? "clip" : "clips"}
                                                </Badge>
                                            )}
                                        </h3>
                                    </div>
                                </div>

                                {loadingLocalClips ? (
                                    <div className="flex items-center justify-center p-8 text-muted-foreground text-sm">
                                        <Loader2 className="h-5 w-5 animate-spin mr-2" /> Loading clips...
                                    </div>
                                ) : localUserClips.length > 0 ? (
                                    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                                        {localUserClips.map((clip, index) => (
                                            <div
                                                key={clip.filename}
                                                className="flex flex-col gap-3 rounded-xl border p-4 bg-card shadow-sm hover:shadow-md transition-shadow"
                                            >
                                                <div className="relative aspect-[9/16] max-h-[380px] w-full overflow-hidden rounded-lg bg-black flex items-center justify-center">
                                                    <video
                                                        src={clip.stream_url}
                                                        controls
                                                        preload="metadata"
                                                        playsInline
                                                        className="h-full w-full object-contain"
                                                    />
                                                    <div className="absolute top-2 left-2 flex gap-1 pointer-events-none">
                                                        <Badge className="bg-blue-600 text-white font-bold text-xs shadow-md">
                                                            #{clip.rank ?? (index + 1)}
                                                        </Badge>
                                                        {clip.aspect_ratio === "9:16" && (
                                                            <Badge variant="secondary" className="bg-black/75 text-white text-[10px]">
                                                                9:16
                                                            </Badge>
                                                        )}
                                                        {clip.subtitles && (
                                                            <Badge variant="secondary" className="bg-amber-500/90 text-white text-[10px]">
                                                                Captions
                                                            </Badge>
                                                        )}
                                                    </div>
                                                </div>
                                                <div className="space-y-1">
                                                    <div className="flex items-center justify-between">
                                                        <span className="text-xs font-semibold truncate max-w-[180px]" title={clip.filename}>
                                                            {clip.filename}
                                                        </span>
                                                        <span className="text-xs font-mono text-muted-foreground">
                                                            {clip.duration ? `${clip.duration.toFixed(1)}s` : ""}
                                                        </span>
                                                    </div>
                                                    <p className="text-[11px] text-muted-foreground">
                                                        {clip.mter_event_id ? `Event ${clip.mter_event_id} • ` : ""}
                                                        {clip.score !== undefined ? `Score: ${clip.score.toFixed(3)}` : `Saved ${new Date(clip.created_at * 1000).toLocaleTimeString()}`}
                                                    </p>
                                                </div>
                                                <div className="pt-1">
                                                    <Button asChild size="sm" variant="default" className="w-full bg-blue-600 hover:bg-blue-700 font-medium">
                                                        <a href={clip.download_url} download={clip.filename}>
                                                            <Download className="h-4 w-4 mr-1.5" /> Download Clip
                                                        </a>
                                                    </Button>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <div className="rounded-lg border border-dashed p-8 text-center text-muted-foreground text-xs space-y-2">
                                        <Sparkles className="h-8 w-8 mx-auto text-slate-300" />
                                        <p className="font-medium text-slate-600">No clips generated yet for this account.</p>
                                        <p className="text-slate-400">
                                            Upload a video in the ClipSense Studio tab to extract highlight clips.
                                        </p>
                                    </div>
                                )}
                            </div>
                        </CardContent>
                    </Card>
                </TabsContent>
            </Tabs>
        </div>
    );
}