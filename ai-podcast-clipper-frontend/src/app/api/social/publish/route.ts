import { NextResponse } from "next/server";
import { auth } from "~/server/auth";
import { inngest } from "~/inngest/client";

export async function POST(req: Request) {
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return new NextResponse("Unauthorized", { status: 401 });
    }

    // Replace your current `const body = await req.json();` with this:
    const body = (await req.json()) as {
      clipId: string;
      platforms: { youtube: boolean; instagram: boolean; facebook: boolean; 
        // x:boolean; 
      };
      ytTitle: string;
      ytDesc: string;
      instaCaption: string;
      fbCaption: string;
      // xCaption: string;
      scheduledTime: string | null;
    };

    const { clipId, platforms, ytTitle, ytDesc, instaCaption, fbCaption, 
      // xCaption, 
      scheduledTime } = body;

    if (!clipId || !platforms) {
      return new NextResponse("Missing required fields", { status: 400 });
    }

    // Trigger the background upload job with the new specific text fields
    await inngest.send({
      name: "publish-social-video",
      data: {
        userId: session.user.id,
        clipId,
        platforms,
        ytTitle,
        ytDesc,
        instaCaption,
        fbCaption,
        // xCaption,
        scheduledTime,
      },
    });

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Publishing error:", error);
    return new NextResponse("Internal Server Error", { status: 500 });
  }
}