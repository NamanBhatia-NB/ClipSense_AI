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
        },
    });

    // Redirect to terms acceptance if not accepted
    if (!userData.termsAccepted) {
        redirect("/accept-terms");
    }

    return (
        <DashboardClient
            userCredits={userData.credits}
            userId={session.user.id}
        />
    );
}