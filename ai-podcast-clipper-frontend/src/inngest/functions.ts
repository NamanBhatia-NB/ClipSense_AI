import { env } from "~/env";
import { inngest } from "./client";
import { db } from "~/server/db";
import { GetObjectCommand, ListObjectsV2Command, S3Client } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { google } from "googleapis";
import { Readable } from "stream";
// import { TwitterApi } from "twitter-api-v2";

export const processVideo = inngest.createFunction(
  {
    id: "process-video",
    retries: 0,
    timeouts: {
      start: "30m",
    },
    concurrency: {
      limit: 1,
      key: "event.data.userId",
    },
  },
  { event: "process-video-events" },
  async ({ event, step }) => {
    const { uploadedFileId, clipSettings } = event.data as {
      uploadedFileId: string;
      userId: string;
      clipSettings: {
        mode: "auto" | "manual";
        requestedClips?: number;
        userPrompt?: string;
      };
    };

    try {
      const { userId, credits, s3Key } = await step.run(
        "check-credits",
        async () => {
          const uploadedFile = await db.uploadedFile.findUniqueOrThrow({
            where: {
              id: uploadedFileId,
            },
            select: {
              user: {
                select: {
                  id: true,
                  credits: true,
                },
              },
              s3Key: true,
            },
          });

          return {
            userId: uploadedFile.user.id,
            credits: uploadedFile.user.credits,
            s3Key: uploadedFile.s3Key,
          };
        },
      );

      if (credits > 0) {
        await step.run("set-status-processing", async () => {
          await db.uploadedFile.update({
            where: {
              id: uploadedFileId,
            },
            data: {
              status: "processing",
            },
          });
        });

        const backendResponse = await step.fetch(env.PROCESS_VIDEO_ENDPOINT, {
          method: "POST",
          body: JSON.stringify({
            s3_key: s3Key,
            mode: clipSettings.mode,
            requested_clips: clipSettings.requestedClips,
            user_credits: credits,
            user_prompt: clipSettings.userPrompt ?? null,
          }),
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${env.PROCESS_VIDEO_ENDPOINT_AUTH}`,
          },
        });

        const backendResult = await backendResponse.json() as unknown as { clip_metadata?: Array<{ s3_key: string; title: string; description: string }> }
        const clipMetadata = backendResult.clip_metadata ?? [];

        const { clipsFound } = await step.run(
          "create-clips-in-db",
          async () => {
            const folderPrefix = s3Key.split("/")[0]!;

            const allKeys = await listS3ObjectsByPrefix(folderPrefix);

            const clipKeys = allKeys.filter(
              (key): key is string =>
                key !== undefined && !key.endsWith("original.mp4"),
            );

            if (clipKeys.length > 0) {
              // Create clips with title and description from backend
              for (const clipKey of clipKeys) {
                const metadata = clipMetadata.find(m => m.s3_key === clipKey);
                await db.clip.create({
                  data: {
                    s3Key: clipKey,
                    uploadedFileId,
                    userId,
                    title: metadata?.title ?? null,
                    description: metadata?.description ?? null,
                  },
                });
              }
            }

            return { clipsFound: clipKeys.length };
          },
        );

        await step.run("deduct-credits", async () => {
          await db.user.update({
            where: {
              id: userId,
            },
            data: {
              credits: {
                decrement: Math.min(credits, clipsFound),
              },
            },
          });
        });

        await step.run("set-status-final", async () => {
          // If clipsFound is 0, status becomes "no_match", otherwise "processed"
          const finalStatus = clipsFound > 0 ? "processed" : "no_match";

          await db.uploadedFile.update({
            where: {
              id: uploadedFileId,
            },
            data: {
              status: finalStatus,
            },
          });
        });

        // Auto Publish Scheduling Logic
        if (clipsFound > 0) {
          await step.run("schedule-auto-posts", async () => {

            // 100% LINT-SAFE CAST: Tells ESLint exactly what data to expect
            const file = (await db.uploadedFile.findUnique({
              where: { id: uploadedFileId },
              include: { clips: { orderBy: { createdAt: 'asc' } } }
            })) as unknown as {
              autoPublish: boolean;
              publishMode: string | null;
              publishPlatforms: {
                youtube?: boolean; instagram?: boolean; facebook?: boolean;
                // x?: boolean;
              } | null;
              publishTimeSlots: string[] | null;
              startDate: Date | string | null;
              ytSuffix: string | null;
              instaSuffix: string | null;
              fbSuffix: string | null;
              // xSuffix: string | null;
              clips: Array<{
                id: string;
                title: string | null;
                description: string | null;
              }>;
            } | null;

            if (!file?.autoPublish) return;

            const platforms = file.publishPlatforms;
            if (!platforms) return;

            const timeSlots = file.publishTimeSlots ?? [];
            if (timeSlots.length === 0) return;

            // Loop through all generated clips to assign them their designated time
            for (let i = 0; i < file.clips.length; i++) {
              const clip = file.clips[i]!;
              let scheduledTime: Date | null = null;

              // Date Math: Is it a Pattern (HH:MM) or Exact (ISO String)?
              const isExact = timeSlots[0]?.includes("T");

              if (isExact) {
                const exactTimeStr = timeSlots[i];
                scheduledTime = exactTimeStr ? new Date(exactTimeStr) : null;
              } else if (file.startDate) {
                const baseDate = new Date(file.startDate);
                const dayOffset = Math.floor(i / timeSlots.length);
                const slotIndex = i % timeSlots.length;

                const timeString = timeSlots[slotIndex]!;
                const [hours, minutes] = timeString.split(":").map(Number);

                scheduledTime = new Date(baseDate);
                scheduledTime.setDate(scheduledTime.getDate() + dayOffset);
                scheduledTime.setHours(hours ?? 0, minutes ?? 0, 0, 0);
              }

              // Text Formatting (Append Suffixes safely)
              const ytTitle = clip.title ?? "";
              const ytDesc = `${clip.description ?? ""}\n\n${file.ytSuffix ?? ""}`.trim();
              const instaCaption = `${clip.title ?? ""}\n\n${clip.description ?? ""}\n\n${file.instaSuffix ?? ""}`.trim();
              const fbCaption = `${clip.title ?? ""}\n\n${clip.description ?? ""}\n\n${file.fbSuffix ?? ""}`.trim();
              // const xCaption = `${clip.title ?? ""}\n\n${clip.description ?? ""}\n\n${file.xSuffix ?? ""}`.trim();

              if (file.publishMode === "direct") {
                await inngest.send({
                  name: "publish-social-video",
                  data: {
                    userId,
                    clipId: clip.id,
                    platforms: {
                      youtube: platforms.youtube ?? false,
                      instagram: platforms.instagram ?? false,
                      facebook: platforms.facebook ?? false,
                      // x: platforms.x ?? false,
                    },
                    ytTitle,
                    ytDesc,
                    instaCaption,
                    fbCaption,
                    // xCaption,
                    scheduledTime: scheduledTime ? scheduledTime.toISOString() : null,
                  },
                });
              } else {
                // Strictly type the array so ESLint doesn't panic on the loop
                const draftData: Array<{ platform: string; status: string }> = [];
                if (platforms.youtube) draftData.push({ platform: "youtube", status: "draft" });
                if (platforms.instagram) draftData.push({ platform: "instagram", status: "draft" });
                if (platforms.facebook) draftData.push({ platform: "facebook", status: "draft" });
                // if (platforms.x) draftData.push({ platform: "x", status: "draft" });

                for (const draft of draftData) {
                  await db.socialPost.create({
                    data: {
                      userId,
                      clipId: clip.id,
                      platform: draft.platform,
                      status: draft.status,
                      scheduledFor: scheduledTime,
                    }
                  });
                }
              }
            }
          });
        }
      } else {
        await step.run("set-status-no-credits", async () => {
          await db.uploadedFile.update({
            where: {
              id: uploadedFileId,
            },
            data: {
              status: "no credits",
            },
          });
        });
      }
    } catch {
      await db.uploadedFile.update({
        where: {
          id: uploadedFileId,
        },
        data: {
          status: "failed",
        },
      });
    }
  },
);

async function listS3ObjectsByPrefix(prefix: string) {
  const s3Client = new S3Client({
    region: env.AWS_REGION,
    credentials: {
      accessKeyId: env.AWS_ACCESS_KEY_ID,
      secretAccessKey: env.AWS_SECRET_ACCESS_KEY,
    },
  });

  const listCommand = new ListObjectsV2Command({
    Bucket: env.S3_BUCKET_NAME,
    Prefix: prefix,
  });

  const response = await s3Client.send(listCommand);
  return response.Contents?.map((item) => item.Key).filter(Boolean) ?? [];
}

export const publishSocialVideo = inngest.createFunction(
  {
    id: "publish-social-video",
    retries: 1,
    timeouts: {
      start: "5m",
    },
  },
  { event: "publish-social-video" },
  async ({ event, step }) => {
    const { userId, clipId, platforms, ytTitle, ytDesc, instaCaption, fbCaption,
      // xCaption,
      scheduledTime } = event.data;

    const { clip, googleAccount, facebookAccount } = await step.run("fetch-data", async () => {
      const clipRecord = await db.clip.findUniqueOrThrow({ where: { id: clipId } });
      const accounts = await db.account.findMany({ where: { userId } });

      return {
        clip: clipRecord,
        googleAccount: accounts.find(a => a.provider === "google"),
        facebookAccount: accounts.find(a => a.provider === "facebook"),
        // xAccount: accounts.find(a => a.provider === "twitter"),
      };
    });

    const initialVideoUrl = await step.run("get-s3-url", async () => {
      const s3Client = new S3Client({
        region: env.AWS_REGION,
        credentials: {
          accessKeyId: env.AWS_ACCESS_KEY_ID,
          secretAccessKey: env.AWS_SECRET_ACCESS_KEY,
        },
      });
      const command = new GetObjectCommand({ Bucket: env.S3_BUCKET_NAME, Key: clip.s3Key });
      return await getSignedUrl(s3Client, command, { expiresIn: 3600 });
    });

    if (platforms.facebook || platforms.instagram) {
      await step.run("upgrade-meta-token", async () => {
        if (!facebookAccount?.access_token) return;

        const exchangeUrl = `https://graph.facebook.com/v19.0/oauth/access_token?grant_type=fb_exchange_token&client_id=${env.AUTH_FACEBOOK_ID}&client_secret=${env.AUTH_FACEBOOK_SECRET}&fb_exchange_token=${facebookAccount.access_token}`;

        const res = await fetch(exchangeUrl, { cache: "no-store" });
        const data = (await res.json()) as unknown as { access_token?: string; error?: unknown };

        // If we got a 60-day token, save it to the DB so FB and IG can use it!
        if (data.access_token) {
          await db.account.update({
            where: { id: facebookAccount.id },
            data: { access_token: data.access_token }
          });
        }
      });
    }

    // YOUTUBE UPLOAD
    if (platforms.youtube) {
      await step.run("publish-youtube", async () => {

        // 1. Ensure we actually have a refresh token saved in the database
        if (!googleAccount?.refresh_token) {
          throw new Error("No Google refresh token found. You must remove the app from your Google Account Security settings and relink.");
        }

        const oauth2Client = new google.auth.OAuth2(env.AUTH_GOOGLE_ID, env.AUTH_GOOGLE_SECRET);

        // 2. THE FIX: Only pass the refresh_token.
        oauth2Client.setCredentials({
          refresh_token: googleAccount.refresh_token,
        });

        const youtube = google.youtube({ version: "v3", auth: oauth2Client });

        const videoResponse = await fetch(initialVideoUrl);
        const videoBuffer = await videoResponse.arrayBuffer();

        const readableStream = new Readable();
        readableStream.push(Buffer.from(videoBuffer));
        readableStream.push(null);

        const res = await youtube.videos.insert({
          part: ["snippet", "status"],
          requestBody: {
            snippet: {
              title: ytTitle,
              description: ytDesc,
              tags: ["Shorts", "Podcast"],
              categoryId: "22",
            },
            status: {
              privacyStatus: scheduledTime ? "private" : "public",
              publishAt: scheduledTime ? new Date(scheduledTime).toISOString() : undefined,
              selfDeclaredMadeForKids: false,
            },
          },
          media: {
            body: readableStream,
          },
        });

        await db.socialPost.create({
          data: {
            clipId, userId, platform: "youtube",
            status: scheduledTime ? "scheduled" : "published",
            externalPostId: res.data.id,
            scheduledFor: scheduledTime ? new Date(scheduledTime) : null,
          }
        });
      });
    }

    // FACEBOOK UPLOAD
    if (platforms.facebook) {
      await step.run("publish-facebook", async () => {
        // Fetch the newly upgraded 60-day token
        const freshFbAccount = await db.account.findUnique({ where: { id: facebookAccount!.id } });
        if (!freshFbAccount?.access_token) throw new Error("No Meta account connected");

        const fbToken = freshFbAccount.access_token;
        const pagesRes = await fetch(`https://graph.facebook.com/v19.0/me/accounts?access_token=${fbToken}`, { cache: "no-store" });

        // 100% Lint-Safe Cast
        const pagesData = (await pagesRes.json()) as unknown as { data?: Array<{ id: string; access_token: string }>; error?: unknown };

        if (!pagesData.data || pagesData.data.length === 0) {
          throw new Error(`Meta API returned no pages. Raw response: ${JSON.stringify(pagesData)}`);
        }

        const page = pagesData.data[0]!;
        const pageId = page.id;
        const pageToken = page.access_token;

        const fbParams = new URLSearchParams({
          file_url: initialVideoUrl,
          description: fbCaption,
          access_token: pageToken,
        });

        if (scheduledTime) {
          fbParams.append("published", "false");
          fbParams.append("scheduled_publish_time", Math.floor(new Date(scheduledTime).getTime() / 1000).toString());
        }

        const fbUpload = await fetch(`https://graph.facebook.com/v19.0/${pageId}/videos`, {
          method: 'POST', body: fbParams, cache: "no-store"
        });

        // 100% Lint-Safe Cast
        const fbResult = (await fbUpload.json()) as unknown as { id?: string; error?: unknown };

        if (fbResult.error) {
          throw new Error(`FB Upload Error: ${JSON.stringify(fbResult.error)}`);
        }

        await db.socialPost.create({
          data: {
            clipId, userId, platform: "facebook",
            status: scheduledTime ? "scheduled" : "published",
            externalPostId: fbResult.id,
            scheduledFor: scheduledTime ? new Date(scheduledTime) : null,
          }
        });
      });
    }

    // INSTAGRAM UPLOAD
    if (platforms.instagram) {
      if (scheduledTime) {
        await step.sleepUntil("wait-for-ig-schedule", new Date(scheduledTime));
      }

      const igState = await step.run("create-ig-container", async () => {
        // BUG FIX 1: Fetch fresh DB credentials in case they changed during sleep
        const freshAccount = await db.account.findFirst({ where: { userId, provider: "facebook" } });
        if (!freshAccount?.access_token) throw new Error("No Meta account connected");
        const fbToken = freshAccount.access_token;

        // BUG FIX 2: Generate a fresh 1-hour S3 link (the old one is expired!)
        const s3Client = new S3Client({
          region: env.AWS_REGION,
          credentials: { accessKeyId: env.AWS_ACCESS_KEY_ID, secretAccessKey: env.AWS_SECRET_ACCESS_KEY },
        });
        const freshVideoUrl = await getSignedUrl(s3Client, new GetObjectCommand({ Bucket: env.S3_BUCKET_NAME, Key: clip.s3Key }), { expiresIn: 3600 });

        const pagesRes = await fetch(`https://graph.facebook.com/v19.0/me/accounts?access_token=${fbToken}`, { cache: "no-store" });
        const pagesData = (await pagesRes.json()) as unknown as { data?: Array<{ id: string }>; error?: unknown };

        // BUG FIX 3: Stop hiding the real Meta error!
        if (pagesData.error) {
          throw new Error(`Meta API Token Error (IG): ${JSON.stringify(pagesData.error)}`);
        }

        const pageId = pagesData.data?.[0]?.id;
        if (!pageId) throw new Error("No Facebook page found. Make sure Page permissions are granted.");

        const igRes = await fetch(`https://graph.facebook.com/v19.0/${pageId}?fields=instagram_business_account&access_token=${fbToken}`, { cache: "no-store" });
        const igData = (await igRes.json()) as unknown as { instagram_business_account?: { id: string }; error?: unknown };
        const igId = igData.instagram_business_account?.id;

        if (!igId) throw new Error("No Instagram Professional account connected to this Facebook Page");

        const igParams = new URLSearchParams({
          video_url: freshVideoUrl,
          media_type: "REELS",
          caption: instaCaption,
          access_token: fbToken,
        });

        const igUpload = await fetch(`https://graph.facebook.com/v19.0/${igId}/media`, { method: 'POST', body: igParams, cache: "no-store" });
        const igResult = (await igUpload.json()) as unknown as { id?: string; error?: unknown };

        if (igResult.error) throw new Error(`IG Upload Error: ${JSON.stringify(igResult.error)}`);

        return { creationId: igResult.id, igId, fbToken };
      });

      await step.sleep("wait-for-ig-processing", "30s");

      await step.run("publish-ig-container", async () => {
        const publishParams = new URLSearchParams({
          creation_id: igState.creationId ?? "",
          access_token: igState.fbToken,
        });

        const igPublish = await fetch(`https://graph.facebook.com/v19.0/${igState.igId}/media_publish`, {
          method: 'POST',
          body: publishParams,
          cache: "no-store"
        });

        const publishData = (await igPublish.json()) as unknown as { id?: string; error?: unknown };

        if (publishData.error) {
          throw new Error(`IG Publish Error: ${JSON.stringify(publishData.error)}`);
        }

        await db.socialPost.create({
          data: {
            clipId, userId, platform: "instagram",
            status: "published",
            externalPostId: publishData.id,
            scheduledFor: scheduledTime ? new Date(scheduledTime) : null,
          }
        });
      });
    }

    /*  
      if (platforms.x) {
        await step.run("publish-x", async () => {
          const xAccount = (await db.account.findMany({ where: { userId } }))
            .find(a => a.provider === "twitter");
  
          if (!xAccount?.access_token || !xAccount?.refresh_token) {
            throw new Error("No X account connected or missing refresh token");
          }
  
          const client = new TwitterApi({
            clientId: env.AUTH_TWITTER_ID,
            clientSecret: env.AUTH_TWITTER_SECRET,
          });
  
          // 1. Get the new tokens from Twitter
          const {
            client: refreshedClient,
            accessToken,
            refreshToken: newRefreshToken
          } = await client.refreshOAuth2Token(xAccount.refresh_token);
  
          // 2. CRITICAL BUG FIX: Save the new rotating tokens back to the database!
          if (newRefreshToken) {
            await db.account.update({
              where: { id: xAccount.id },
              data: {
                access_token: accessToken,
                refresh_token: newRefreshToken,
              }
            });
          }
  
          // 3. Download video
          const videoResponse = await fetch(initialVideoUrl);
          const videoBuffer = Buffer.from(await videoResponse.arrayBuffer());
  
          let mediaId;
          try {
            // 4. Upload Media
            mediaId = await refreshedClient.v1.uploadMedia(videoBuffer, {
              mimeType: 'video/mp4',
              target: 'tweet',
              longVideo: true,
            });
          } catch (error: any) {
            console.error("X API REJECTION DATA:", JSON.stringify(error?.data || error));
            throw new Error(`X Media Upload Rejected: ${error?.data?.detail || error.message}`);
          }
  
          // 5. Prepare Caption
          // const fullCaption = xCaption || `${ytTitle}\n\n${ytDesc}`.trim();
          // const xText = fullCaption.length > 280 ? fullCaption.substring(0, 277) + "..." : fullCaption;
  
          try {
            // 6. Post Tweet
            const tweet = await refreshedClient.v2.tweet({
              text: xText,
              media: { media_ids: [mediaId] }
            });
  
            await db.socialPost.create({
              data: {
                clipId, userId, platform: "x",
                status: scheduledTime ? "scheduled" : "published",
                externalPostId: tweet.data.id,
                scheduledFor: scheduledTime ? new Date(scheduledTime) : null,
              }
            });
          } catch (error: any) {
            console.error("X TWEET REJECTION DATA:", JSON.stringify(error?.data || error));
            throw new Error(`X Tweet Rejected: ${error?.data?.detail || error.message}`);
          }
        });
      }
    */
    return { success: true };
  }
);
