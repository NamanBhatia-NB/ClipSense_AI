"use client";

import { Check, Copy, Download, Loader2, Send, Clock, X, Youtube, Instagram, Lock, Facebook } from "lucide-react";
import { useEffect, useState, type Dispatch, type SetStateAction } from "react";
import { toast } from "sonner";
import { signIn } from "next-auth/react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Textarea } from "./ui/textarea";
import { Switch } from "./ui/switch";
import { Label } from "./ui/label";
import { approveDraftPost } from "~/actions/generation";
import { Badge } from "./ui/badge";

const PLATFORM_LIMITS = {
    youtubeTitle: 100,
    youtubeDesc: 5000,
    instagram: 2200,
    facebook: 2200,
    x: 280
};

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

// 1. We define the type for our connections so we can pass it around
export type SocialConnections = {
    youtubeLinked: boolean;
    metaLinked: boolean;
    // xLinked: boolean;
    canConnectMeta: boolean;
    hasRequestedInvite: boolean;
    isLoading: boolean;
};

function PublishModal({
    clip,
    onClose,
    connections,
    setConnections
}: {
    clip: ClipWithUrl;
    onClose: () => void;
    connections: SocialConnections;
    setConnections: Dispatch<SetStateAction<SocialConnections>>;
}) {
    // We instantly set the checkboxes based on what is already linked!
    const [platforms, setPlatforms] = useState({
        instagram: connections.metaLinked,
        facebook: connections.metaLinked,
        youtube: connections.youtubeLinked,
        // x: connections.xLinked,
    });

    // --- TEXT STATES ---
    const [useSeparateText, setUseSeparateText] = useState(false);
    const [globalTitle, setGlobalTitle] = useState(clip.title ?? "");
    const [globalCaption, setGlobalCaption] = useState(`${clip.title ?? ""}\n\n${clip.description ?? ""}`);
    const [ytTitle, setYtTitle] = useState(clip.title ?? "");
    const [ytDesc, setYtDesc] = useState(clip.description ?? "");
    const [instaCaption, setInstaCaption] = useState(`${clip.title ?? ""}\n\n${clip.description ?? ""}`);
    const [fbCaption, setFbCaption] = useState(`${clip.title ?? ""}\n\n${clip.description ?? ""}`);
    // const [xCaption, setXCaption] = useState(`${clip.title ?? ""}\n\n${clip.description ?? ""}`);

    const [activeTab, setActiveTab] = useState<"youtube" | "instagram" | "facebook" 
    // | "x"
    >
    ("youtube");
    const [isScheduling, setIsScheduling] = useState(false);
    const [scheduleDate, setScheduleDate] = useState("");
    const [isPublishing, setIsPublishing] = useState(false);

    // Auto-switch tabs if the user unchecks the currently active tab
    useEffect(() => {
        if (!platforms[activeTab]) {
            if (platforms.instagram) setActiveTab("instagram");
            else if (platforms.facebook) setActiveTab("facebook");
            else if (platforms.youtube) setActiveTab("youtube");
            // else if (platforms.x) setActiveTab("x");
        }
    }, [platforms, activeTab]);

    const handleRequestInvite = async () => {
        const toastId = toast.loading("Requesting access...");
        try {
            const res = await fetch("/api/social/request-invite", { method: "POST" });
            if (res.ok) {
                toast.success("Invite requested! We'll notify you when approved.", { id: toastId });
                // We update the GLOBAL state so all other clips know about this request!
                setConnections(prev => ({ ...prev, hasRequestedInvite: true }));
            } else {
                toast.error("Failed to request invite.", { id: toastId });
            }
        } catch (e) {
            toast.error("Something went wrong.", { id: toastId });
        }
    };

    const handlePublish = async () => {
        if (!platforms.youtube && !platforms.instagram && !platforms.facebook  
            // && !platforms.x
        ) {
            toast.error("Please select at least one connected platform.");
            return;
        }

        if (isScheduling && !scheduleDate) {
            toast.error("Please select a date and time to schedule.");
            return;
        }

        setIsPublishing(true);
        const toastId = toast.loading(isScheduling ? "Scheduling post..." : "Initiating upload...");

        try {
            const response = await fetch("/api/social/publish", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    clipId: clip.id,
                    platforms,
                    ytTitle: useSeparateText ? ytTitle : globalTitle,
                    ytDesc: useSeparateText ? ytDesc : globalCaption,
                    instaCaption: useSeparateText ? instaCaption : globalCaption,
                    fbCaption: useSeparateText ? fbCaption : globalCaption,
                    // xCaption: useSeparateText ? xCaption : globalCaption,
                    scheduledTime: isScheduling ? new Date(scheduleDate).toISOString() : null
                }),
            });

            if (response.ok) {
                toast.success(isScheduling ? "Clip scheduled successfully!" : "Uploading to platforms!", { id: toastId });
                onClose();
            } else {
                toast.error("Failed to start upload process.", { id: toastId });
                setIsPublishing(false);
            }
        } catch (error) {
            toast.error("An error occurred.", { id: toastId });
            setIsPublishing(false);
        }
    };

    const hasPlatformsSelected = (platforms.instagram && connections.metaLinked) ||
        (platforms.facebook && connections.metaLinked) ||
        (platforms.youtube && connections.youtubeLinked) 
        // || (platforms.x && connections.xLinked)
        ;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 backdrop-blur-sm">
            <div className="bg-background rounded-xl shadow-lg border space-y-4 max-w-md w-full p-6 relative max-h-[90vh] overflow-y-auto">

                <button onClick={onClose} className="absolute right-4 top-4 text-muted-foreground hover:text-foreground">
                    <X className="w-5 h-5" />
                </button>

                <h3 className="text-lg font-bold">Publish Clip</h3>

                {/* --- SMART ACCOUNT CONNECTION SECTION --- */}
                <div className="bg-muted p-4 rounded-lg text-sm border">
                    <p className="font-semibold mb-2">Connected Accounts</p>

                    {connections.isLoading ? (
                        <div className="flex items-center text-muted-foreground text-xs">
                            <Loader2 className="w-3 h-3 animate-spin mr-2" /> Checking connections...
                        </div>
                    ) : (
                        <div className="flex gap-2 items-center">
                            {connections.youtubeLinked ? (
                                <div className="flex-1 flex items-center justify-center gap-2 border bg-green-50 text-green-700 rounded-md p-2 text-xs font-medium text-center">
                                    <Check className="w-4 h-4" /> YouTube Ready
                                </div>
                            ) : (
                                <Button size="sm" variant="outline" onClick={() => signIn("google")} className="flex-1 bg-white hover:bg-red-50 hover:text-red-600 hover:border-red-200">
                                    <Youtube className="w-4 h-4 mr-2 text-red-500" /> Link YouTube
                                </Button>
                            )}

                            {connections.metaLinked ? (
                                <div className="flex-1 flex items-center justify-center gap-2 border bg-green-50 text-green-700 rounded-md p-2 text-xs font-medium text-center">
                                    <Check className="w-4 h-4" /> Meta Ready
                                </div>
                            ) : connections.canConnectMeta ? (
                                <Button size="sm" variant="outline" onClick={() => signIn("facebook")} className="flex-1 bg-white hover:bg-pink-50 hover:text-pink-600 hover:border-pink-200">
                                    <Instagram className="w-4 h-4 mr-2 text-pink-500" /> Link Meta
                                </Button>
                            ) : connections.hasRequestedInvite ? (
                                <Button size="sm" variant="outline" disabled className="flex-1 bg-slate-50 opacity-70">
                                    <Clock className="w-4 h-4 mr-2 text-slate-400" /> Invite Pending
                                </Button>
                            ) : (
                                <Button size="sm" variant="outline" onClick={handleRequestInvite} className="flex-1 bg-white hover:bg-slate-50">
                                    <Lock className="w-4 h-4 mr-2 text-slate-400" /> Request Meta Access
                                </Button>
                            )}
                            {/* {connections.xLinked ? (
                                <div className="flex-1 flex items-center justify-center gap-2 border bg-green-50 text-green-700 rounded-md p-2 text-xs font-medium">
                                    <Check className="w-4 h-4" /> X Ready
                                </div>
                            ) : (
                                <Button size="sm" variant="outline" onClick={() => signIn("twitter")} className="flex-1 bg-white hover:bg-slate-100 hover:text-slate-900 hover:border-slate-300">
                                    <img src='/x.svg' className="w-4 h-4 mr-2 text-slate-700" /> Link X
                                </Button>
                            )} */}
                        </div>
                    )}
                </div>

                {/* --- PLATFORM SELECTORS --- */}
                <div className="flex gap-4 pt-2 border-b pb-4">
                    <label className={`flex items-center gap-2 text-sm font-medium ${!connections.youtubeLinked ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}>
                        <input
                            type="checkbox"
                            className="rounded cursor-pointer disabled:cursor-not-allowed"
                            checked={platforms.youtube && connections.youtubeLinked}
                            disabled={!connections.youtubeLinked}
                            onChange={(e) => setPlatforms({ ...platforms, youtube: e.target.checked })}
                        />
                        YouTube
                    </label>
                    <label className={`flex items-center gap-2 text-sm font-medium ${!connections.metaLinked ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}>
                        <input
                            type="checkbox"
                            className="rounded cursor-pointer disabled:cursor-not-allowed"
                            checked={platforms.instagram && connections.metaLinked}
                            disabled={!connections.metaLinked}
                            onChange={(e) => setPlatforms({ ...platforms, instagram: e.target.checked })}
                        />
                        Instagram
                    </label>
                    <label className={`flex items-center gap-2 text-sm font-medium ${!connections.metaLinked ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}>
                        <input
                            type="checkbox"
                            className="rounded cursor-pointer disabled:cursor-not-allowed"
                            checked={platforms.facebook && connections.metaLinked}
                            disabled={!connections.metaLinked}
                            onChange={(e) => setPlatforms({ ...platforms, facebook: e.target.checked })}
                        />
                        Facebook
                    </label>
                    
                    {/* <label className={`flex items-center gap-2 text-sm font-medium ${!connections.xLinked ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}>
                        <input
                            type="checkbox"
                            className="rounded cursor-pointer disabled:cursor-not-allowed"
                            checked={platforms.x && connections.xLinked}
                            disabled={!connections.xLinked}
                            onChange={(e) => setPlatforms({ ...platforms, x: e.target.checked })}
                        />
                        X
                    </label> */}
                </div>

                {/* --- CONTENT INPUTS --- */}
                {hasPlatformsSelected && (
                    <div className="space-y-4 pt-2">
                        <div className="flex items-center justify-between bg-slate-50 p-3 rounded-lg border">
                            <Label className="text-sm font-medium cursor-pointer">Customize text for each platform</Label>
                            <Switch checked={useSeparateText} onCheckedChange={setUseSeparateText} />
                        </div>

                        {!useSeparateText ? (
                            <div className="space-y-3 animate-in fade-in zoom-in duration-200">
                                {platforms.youtube && (
                                    <div className="space-y-1">
                                        <Label className="text-xs">Shorts Title (YouTube Only)</Label>
                                        <Input value={globalTitle} onChange={(e) => setGlobalTitle(e.target.value)} placeholder="Catchy title..." />
                                    </div>
                                )}
                                <div className="space-y-1">
                                    <Label className="text-xs">Caption & Description</Label>
                                    <Textarea rows={4} value={globalCaption} onChange={(e) => setGlobalCaption(e.target.value)} placeholder="Write your caption here..." />
                                </div>
                            </div>
                        ) : (
                            <div className="border rounded-lg overflow-hidden animate-in fade-in slide-in-from-bottom-2 duration-200">
                                <div className="flex border-b bg-muted/50">
                                    {platforms.youtube && connections.youtubeLinked && (
                                        <button onClick={() => setActiveTab("youtube")} className={`flex-1 flex items-center justify-center gap-2 py-2 text-xs font-medium border-b-2 transition-colors ${activeTab === "youtube" ? "border-red-500 text-foreground bg-background" : "border-transparent text-muted-foreground hover:bg-muted"}`}>
                                            <Youtube className="w-3 h-3 text-red-500" /> YouTube
                                        </button>
                                    )}
                                    {platforms.instagram && connections.metaLinked && (
                                        <button onClick={() => setActiveTab("instagram")} className={`flex-1 flex items-center justify-center gap-2 py-2 text-xs font-medium border-b-2 transition-colors ${activeTab === "instagram" ? "border-pink-500 text-foreground bg-background" : "border-transparent text-muted-foreground hover:bg-muted"}`}>
                                            <Instagram className="w-3 h-3 text-pink-500" /> Instagram
                                        </button>
                                    )}
                                    {platforms.facebook && connections.metaLinked && (
                                        <button onClick={() => setActiveTab("facebook")} className={`flex-1 flex items-center justify-center gap-2 py-2 text-xs font-medium border-b-2 transition-colors ${activeTab === "facebook" ? "border-blue-600 text-foreground bg-background" : "border-transparent text-muted-foreground hover:bg-muted"}`}>
                                            <Facebook className="w-3 h-3 text-blue-600" /> Facebook
                                        </button>
                                    )}
                                    {/* {platforms.x && connections.xLinked && (
                                        <button onClick={() => setActiveTab("x")} className={`flex-1 flex items-center justify-center gap-2 py-2 text-xs font-medium border-b-2 transition-colors ${activeTab === "x" ? "border-black-600 text-foreground bg-background" : "border-transparent text-muted-foreground hover:bg-muted"}`}>
                                            <img src='/x.svg' className="w-4 h-4 mr-2 text-slate-700" /> X
                                        </button>
                                    )} */}
                                </div>

                                <div className="p-4 bg-background">
                                    {activeTab === "youtube" && platforms.youtube && connections.youtubeLinked && (
                                        <div className="space-y-3">
                                            <div>
                                                <Label className="text-xs" >YouTube Shorts Title ({ytTitle.length}/{PLATFORM_LIMITS.youtubeTitle})</Label>
                                                <Input
                                                    maxLength={PLATFORM_LIMITS.youtubeTitle}
                                                    value={ytTitle}
                                                    onChange={(e) => setYtTitle(e.target.value)}
                                                />
                                            </div>
                                            <div className="space-y-1">
                                                <Label className="text-xs">YouTube Shorts Description</Label>
                                                <Textarea maxLength={PLATFORM_LIMITS.youtubeDesc} rows={3} value={ytDesc} onChange={(e) => setYtDesc(e.target.value)} />
                                            </div>
                                        </div>
                                    )}
                                    {activeTab === "instagram" && platforms.instagram && connections.metaLinked && (
                                        <div className="space-y-1">
                                            <Label className="text-xs">Instagram Caption</Label>
                                            <Textarea maxLength={PLATFORM_LIMITS.instagram} rows={4} value={instaCaption} onChange={(e) => setInstaCaption(e.target.value)} />
                                        </div>
                                    )}
                                    {activeTab === "facebook" && platforms.facebook && connections.metaLinked && (
                                        <div className="space-y-1">
                                            <Label className="text-xs">Facebook Caption</Label>
                                            <Textarea maxLength={PLATFORM_LIMITS.facebook} rows={4} value={fbCaption} onChange={(e) => setFbCaption(e.target.value)} />
                                        </div>
                                    )}
                                    {/* {activeTab == "x" && platforms.x && connections.xLinked && (
                                        <div className="space-y-1">
                                            <Label className="text-xs" >X Post ({xCaption.length}/{PLATFORM_LIMITS.x})</Label>
                                            <Textarea
                                                maxLength={PLATFORM_LIMITS.x}
                                                value={xCaption}
                                                onChange={(e) => setXCaption(e.target.value)}
                                            />
                                        </div>
                                    )} */}
                                </div>
                            </div>
                        )}
                    </div>
                )}

                <div className="border-t pt-4 space-y-3">
                    <div className="flex items-center justify-between">
                        <Label className="flex items-center gap-2 cursor-pointer">
                            <Clock className="w-4 h-4" /> Schedule for later
                        </Label>
                        <Switch checked={isScheduling} onCheckedChange={setIsScheduling} />
                    </div>

                    {isScheduling && (
                        <Input
                            type="datetime-local"
                            value={scheduleDate}
                            onChange={(e) => setScheduleDate(e.target.value)}
                            className="w-full"
                        />
                    )}
                </div>

                <Button
                    className="w-full mt-4"
                    onClick={handlePublish}
                    disabled={isPublishing || connections.isLoading || (!hasPlatformsSelected)}
                >
                    {isPublishing ? (
                        <span className="flex items-center gap-2"><Loader2 className="w-4 h-4 animate-spin" /> Processing...</span>
                    ) : isScheduling ? (
                        "Schedule Video"
                    ) : (
                        <span className="flex items-center gap-2"><Send className="w-4 h-4" /> Post Now</span>
                    )}
                </Button>
            </div>
        </div>
    );
}

// 2. We pass connections down through the ClipCard
function ClipCard({
    clip,
    connections,
    setConnections
}: {
    clip: ClipWithUrl;
    connections: SocialConnections;
    setConnections: Dispatch<SetStateAction<SocialConnections>>;
}) {
    const [copiedTitle, setCopiedTitle] = useState(false);
    const [copiedDescription, setCopiedDescription] = useState(false);
    const [showPublishModal, setShowPublishModal] = useState(false);

    const handleDownload = () => {
        const link = document.createElement("a");
        link.href = clip.playUrl;
        link.style.display = "none";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    const handleCopyTitle = async () => {
        if (!clip.title) return;
        await navigator.clipboard.writeText(clip.title);
        setCopiedTitle(true);
        setTimeout(() => setCopiedTitle(false), 2000);
    };

    const handleCopyDescription = async () => {
        if (!clip.description) return;
        await navigator.clipboard.writeText(clip.description);
        setCopiedDescription(true);
        setTimeout(() => setCopiedDescription(false), 2000);
    };

    const [approvingId, setApprovingId] = useState<string | null>(null);

    const handleApprove = async (postId: string) => {
        setApprovingId(postId);
        const toastId = toast.loading("Approving post...");
        try {
            // Import this action at the top of your file!
            await approveDraftPost(postId);
            toast.success("Post approved and scheduled!", { id: toastId });
        } catch (error) {
            toast.error("Failed to approve post", { id: toastId });
        } finally {
            setApprovingId(null);
        }
    };

    return (
        <>
            <div className="flex flex-col gap-3 rounded-lg border p-3">
                <div className="bg-muted rounded-md overflow-hidden aspect-[9/16]">
                    <video src={clip.playUrl} controls preload="metadata" className="h-full w-full object-cover" />
                </div>

                {(clip.title ?? clip.description) && (
                    <div className="space-y-2">
                        {clip.title && (
                            <div className="flex items-start justify-between gap-2">
                                <h3 className="font-bold text-sm line-clamp-2 flex-1">{clip.title}</h3>
                                <Button onClick={handleCopyTitle} variant="ghost" size="sm" className="h-6 px-2 shrink-0">
                                    {copiedTitle ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
                                </Button>
                            </div>
                        )}
                        {clip.description && (
                            <div className="flex items-start justify-between gap-2">
                                <p className="text-xs text-muted-foreground line-clamp-3 flex-1">{clip.description}</p>
                                <Button onClick={handleCopyDescription} variant="ghost" size="sm" className="h-6 px-2 shrink-0">
                                    {copiedDescription ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
                                </Button>
                            </div>
                        )}
                    </div>
                )}

                {/* --- SCHEDULED & DRAFT POSTS --- */}
                {clip.socialPosts && clip.socialPosts.length > 0 && (
                    <div className="mt-3 space-y-2 border-t pt-3">
                        <h4 className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Publishing Status</h4>
                        <div className="space-y-2">
                            {clip.socialPosts.map(post => (
                                <div key={post.id} className="flex items-center justify-between text-sm bg-slate-50 p-2 rounded-md border">
                                    <div className="flex items-center gap-2">
                                        {post.platform === "youtube" && <Youtube className="w-3 h-3 text-red-500" />}
                                        {post.platform === "instagram" && <Instagram className="w-3 h-3 text-pink-500" />}
                                        {post.platform === "facebook" && <Facebook className="w-3 h-3 text-blue-600" />}
                                        {/* {post.platform === "x" && <img src="/x.svg" className="w-3 h-3 text-black" />} */}

                                        <div className="flex flex-col">
                                            <span className="capitalize font-medium text-xs">{post.platform}</span>
                                            {post.scheduledFor && (
                                                <span className="text-[10px] text-muted-foreground">
                                                    {new Date(post.scheduledFor).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
                                                </span>
                                            )}
                                        </div>
                                    </div>

                                    {post.status === "draft" ? (
                                        <Button
                                            size="sm"
                                            className="h-7 text-xs bg-blue-500 hover:bg-blue-600 text-white"
                                            onClick={() => handleApprove(post.id)}
                                            disabled={approvingId === post.id}
                                        >
                                            {approvingId === post.id ? <Loader2 className="w-3 h-3 animate-spin" /> : "Approve"}
                                        </Button>
                                    ) : (
                                        <Badge variant={post.status === "failed" ? "destructive" : "secondary"} className="text-[10px]">
                                            {post.status}
                                        </Badge>
                                    )}
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                <div className="flex gap-2 pt-2 border-t">
                    <Button onClick={handleDownload} variant="outline" size="sm" className="flex-1">
                        <Download className="mr-1.5 h-4 w-4" /> Download
                    </Button>
                    <Button onClick={() => setShowPublishModal(true)} size="sm" className="flex-1 text-white">
                        <Send className="mr-1.5 h-4 w-4" /> Publish
                    </Button>
                </div>
            </div>

            {showPublishModal && (
                <PublishModal
                    clip={clip}
                    onClose={() => setShowPublishModal(false)}
                    connections={connections}
                    setConnections={setConnections}
                />
            )}
        </>
    )
}

// 3. We fetch the API here ONCE when the dashboard loads
export function ClipDisplay({
    clips,
    onLoadMore,
    hasMore,
    isLoading
}: {
    clips: ClipWithUrl[];
    onLoadMore: () => void;
    hasMore: boolean;
    isLoading: boolean;
}) {
    // This state is shared across all clips!
    const [connections, setConnections] = useState<SocialConnections>({
        youtubeLinked: false,
        metaLinked: false,
        // xLinked: false,
        canConnectMeta: false,
        hasRequestedInvite: false,
        isLoading: true // Starts true while fetching
    });

    // Fetches ONE time silently in the background
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
                    // xLinked: data.xLinked ?? false,
                    canConnectMeta: data.canConnectMeta ?? false,
                    hasRequestedInvite: data.hasRequestedInvite ?? false,
                    isLoading: false,
                });

                // If in dashboard-client, you also have this:
                // setPlatforms({ youtube: data.youtubeLinked, instagram: data.metaLinked, facebook: data.metaLinked });
            })
            .catch(() => setConnections(prev => ({ ...prev, isLoading: false })));
    }, []);

    if (clips.length === 0) {
        return <p className="text-muted-foreground p-4 text-center">No clips generated yet.</p>;
    }

    return (
        <div className="space-y-4">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {clips.map((clip) => (
                    <ClipCard
                        key={clip.id}
                        clip={clip}
                        connections={connections}
                        setConnections={setConnections}
                    />
                ))}
            </div>
            {hasMore && (
                <div className="flex justify-center mt-6">
                    <Button onClick={onLoadMore} disabled={isLoading} variant="outline" size="lg">
                        {isLoading ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Loading...</> : "Load More"}
                    </Button>
                </div>
            )}
        </div>
    )
}