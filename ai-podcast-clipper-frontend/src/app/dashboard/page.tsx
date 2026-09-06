"use server";

import { redirect } from "next/navigation";
import { DashboardClient } from "~/components/dashboard-client";
import { auth } from "~/server/auth";
import { db } from "~/server/db";

export default async function DashboardPage() {
    const session = await auth();

    if (!session?.user?.id) {
        redirect("/login");
    }

    const userData = await db.user.findUniqueOrThrow({
        where: { id: session.user.id },
        select: {
            credits: true,
            termsAccepted: true,
            uploadedFiles: {
                where: {
                    uploaded: true
                },
                select: {
                    id: true,
                    s3Key: true,
                    displayName: true,
                    status: true,
                    createdAt: true,
                    _count: {
                        select: {
                            clips: true,
                        },
                    },
                },
            },
            _count: {
                select: {
                    clips: true,
                },
            },
        },
    });

    // Redirect to terms acceptance if not accepted
    if (!userData.termsAccepted) {
        redirect("/accept-terms");
    }

    const formattedFiles = userData.uploadedFiles.map((file) => ({
        id: file.id,
        s3Key: file.s3Key,
        filename: file.displayName ?? "Unknown filename",
        status: file.status,
        clipsCount: file._count.clips,
        createdAt: file.createdAt,
    }));

    return (
        <DashboardClient
            uploadedFiles={formattedFiles}
            totalClipsCount={userData._count.clips}
            userCredits={userData.credits}
        />
    );
}