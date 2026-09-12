import { NextResponse } from "next/server";
import { auth } from "~/server/auth";
import { db } from "~/server/db";

export async function POST(req: Request) { // <-- ADDED req: Request HERE
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return new NextResponse("Unauthorized", { status: 401 });
    }

    // Update the user in the database to record the request
    await db.user.update({
      where: { id: session.user.id },
      data: { metaInviteRequested: new Date() },
    });

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Error requesting invite:", error);
    return new NextResponse("Internal Server Error", { status: 500 });
  }
}