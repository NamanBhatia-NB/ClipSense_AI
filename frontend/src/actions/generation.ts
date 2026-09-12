"use server";

import { GetObjectCommand, S3Client } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { revalidatePath } from "next/cache";
import { env } from "~/env";
import { inngest } from "~/inngest/client";
import { auth } from "~/server/auth";
import { db } from "~/server/db";

export async function processVideo(
  uploadedFileId: string,
  clipSettings?: {
    mode: "auto" | "manual";
    requestedClips?: number;
    userPrompt?: string;
    autoPublish?: boolean;
    publishMode?: string;
    scheduleType?: string;
    startDate?: string;
    ytSuffix?: string;
    instaSuffix?: string;
    fbSuffix?: string;
    // xSuffix?: string;
    publishPlatforms?: { youtube?: boolean; instagram?: boolean; facebook?: boolean } | null;
    publishTimeSlots?: string[] | null;
  }
) {
  const uploadedVideo = await db.uploadedFile.findUniqueOrThrow({
    where: { id: uploadedFileId },
    select: { uploaded: true, id: true, userId: true },
  });

  if (uploadedVideo.uploaded) return;

  await inngest.send({
    name: "process-video-events",
    data: {
      uploadedFileId: uploadedVideo.id,
      userId: uploadedVideo.userId,
      clipSettings: clipSettings ?? { mode: "auto" }
    },
  });

  await db.uploadedFile.update({
    where: { id: uploadedFileId },
    data: {
      uploaded: true,
      userPrompt: clipSettings?.userPrompt ?? null,
      // --- SAVE PUBLISHING CONFIG TO DB ---
      autoPublish: clipSettings?.autoPublish ?? false,
      publishMode: clipSettings?.publishMode ?? null,
      
      publishPlatforms: clipSettings?.publishPlatforms 
        ? {
            youtube: clipSettings.publishPlatforms.youtube ?? false,
            instagram: clipSettings.publishPlatforms.instagram ?? false,
            facebook: clipSettings.publishPlatforms.facebook ?? false,
          }
        : undefined,
        
      publishTimeSlots: clipSettings?.publishTimeSlots ?? undefined,
      
      startDate: clipSettings?.startDate ? new Date(clipSettings.startDate) : null,
      ytSuffix: clipSettings?.ytSuffix ?? null,
      instaSuffix: clipSettings?.instaSuffix ?? null,
      fbSuffix: clipSettings?.fbSuffix ?? null,
      // xSuffix: clipSettings?.xSuffix ?? null,
    },
  });

  revalidatePath("/dashboard");
}

export async function getClipPlayUrl(
  clipId: string,
): Promise<{ success: boolean; url?: string; error?: string }> {
  const session = await auth();
  if (!session?.user?.id) {
    return { success: false, error: "Unauthorized" };
  }

  try {
    const clip = await db.clip.findUniqueOrThrow({
      where: {
        id: clipId,
        userId: session.user.id,
      },
    });

    const s3Client = new S3Client({
      region: env.AWS_REGION,
      credentials: {
        accessKeyId: env.AWS_ACCESS_KEY_ID,
        secretAccessKey: env.AWS_SECRET_ACCESS_KEY,
      },
    });

    const command = new GetObjectCommand({
      Bucket: env.S3_BUCKET_NAME,
      Key: clip.s3Key,
    });

    const signedUrl = await getSignedUrl(s3Client, command, {
      expiresIn: 3600,
    });

    return { success: true, url: signedUrl };
  } catch (error) {
    return { success: false, error: "Failed to generate play URL." };
  }
}

export async function getClipsPaginated(
  page = 0,
  pageSize = 12
): Promise<{
  success: boolean;
  clips?: Array<{
    id: string;
    s3Key: string;
    title: string | null;
    description: string | null;
    createdAt: Date;
    playUrl: string;
  }>;
  hasMore: boolean;
  error?: string;
}> {
  const session = await auth();
  if (!session?.user?.id) {
    return { success: false, hasMore: false, error: "Unauthorized" };
  }

  try {
    const clips = await db.clip.findMany({
      where: {
        userId: session.user.id,
      },
      orderBy: {
        createdAt: "desc",
      },
      skip: page * pageSize,
      take: pageSize + 1,
      select: {
        id: true,
        s3Key: true,
        title: true,
        description: true,
        createdAt: true,
        socialPosts: {
          orderBy: { createdAt: "asc" }
        }
      },
    });

    const hasMore = clips.length > pageSize;
    const clipsToReturn = hasMore ? clips.slice(0, pageSize) : clips;

    const s3Client = new S3Client({
      region: env.AWS_REGION,
      credentials: {
        accessKeyId: env.AWS_ACCESS_KEY_ID,
        secretAccessKey: env.AWS_SECRET_ACCESS_KEY,
      },
    });

    const clipsWithUrls = await Promise.all(
      clipsToReturn.map(async (clip) => {
        const command = new GetObjectCommand({
          Bucket: env.S3_BUCKET_NAME,
          Key: clip.s3Key,
        });

        const playUrl = await getSignedUrl(s3Client, command, {
          expiresIn: 3600,
        });

        return {
          ...clip,
          playUrl,
        };
      })
    );

    return { success: true, clips: clipsWithUrls, hasMore };
  } catch (error) {
    return { success: false, hasMore: false, error: "Failed to fetch clips." };
  }
}

export async function approveDraftPost(postId: string) {
  const session = await auth();
  if (!session?.user?.id) throw new Error("Unauthorized");

  // 1. Fetch the draft post and its parent clip + uploaded file
  const post = await db.socialPost.findUniqueOrThrow({
    where: { id: postId, userId: session.user.id },
    include: {
      clip: {
        include: { uploadedFile: true }
      }
    }
  });

  if (post.status !== "draft") throw new Error("Post is not a draft");

  // 2. Reconstruct the precise captions with the user's saved suffixes
  const file = post.clip.uploadedFile as unknown as {
    ytSuffix?: string | null;
    instaSuffix?: string | null;
    fbSuffix?: string | null;
    // xSuffix?: string | null;
  };

  const ytTitle = post.clip.title ?? "";
  const ytDesc = `${post.clip.description ?? ""}\n\n${file?.ytSuffix ?? ""}`.trim();
  const instaCaption = `${post.clip.title ?? ""}\n\n${post.clip.description ?? ""}\n\n${file?.instaSuffix ?? ""}`.trim();
  const fbCaption = `${post.clip.title ?? ""}\n\n${post.clip.description ?? ""}\n\n${file?.fbSuffix ?? ""}`.trim();
  // const xCaption = `${post.clip.title ?? ""}\n\n${post.clip.description ?? ""}\n\n${file?.xSuffix ?? ""}`.trim();

  // 3. Update the database to lock it in
  await db.socialPost.update({
    where: { id: postId },
    data: { status: post.scheduledFor ? "scheduled" : "published" }
  });

  // 4. Fire the Inngest event for this specific platform!
  await inngest.send({
    name: "publish-social-video",
    data: {
      userId: post.userId,
      clipId: post.clipId,
      platforms: {
        youtube: post.platform === "youtube",
        instagram: post.platform === "instagram",
        facebook: post.platform === "facebook",
        // x: post.platform === "x",
      },
      ytTitle,
      ytDesc,
      instaCaption,
      fbCaption,
      // xCaption,
      scheduledTime: post.scheduledFor ? post.scheduledFor.toISOString() : null,
    },
  });

  revalidatePath("/dashboard");
  return { success: true };
}