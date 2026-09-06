import { NextResponse } from "next/server";
import { auth } from "~/server/auth";
import { db } from "~/server/db";

export async function GET() {
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return new NextResponse("Unauthorized", { status: 401 });
    }

    // ONE database call: Fetch the user, their VIP flag, and their accounts all at once.
    const user = await db.user.findUnique({
      where: { id: session.user.id },
      include: { 
        accounts: {
          select: {
            provider: true,
            scope: true, 
            access_token: true 
          }
        }
      },
    });

    if (!user) {
      return new NextResponse("User not found", { status: 404 });
    }

    // A user is only "YouTube Ready" if they have Google AND the upload scope
    const isYouTubeReady = user.accounts.some(a =>
      a.provider === "google" &&
      a.scope?.includes("youtube.upload")
    );

    // A user is only "Meta Ready" if they have Facebook AND an access token
    const isMetaReady = user.accounts.some(a =>
      a.provider === "facebook" &&
      a.access_token != null
    );

    // const isXReady = user.accounts.some(a =>
    //   a.provider === "twitter" &&
    //   a.access_token != null
    // );

    return NextResponse.json({
      youtubeLinked: isYouTubeReady,
      metaLinked: isMetaReady,
      // xLinked: isXReady,
      canConnectMeta: user.canConnectMeta, // This successfully passes the Prisma VIP flag to your frontend!
      hasRequestedInvite: user.metaInviteRequested !== null,
    });
  } catch (error) {
    console.error("Error fetching social accounts:", error);
    return new NextResponse("Internal Server Error", { status: 500 });
  }
}