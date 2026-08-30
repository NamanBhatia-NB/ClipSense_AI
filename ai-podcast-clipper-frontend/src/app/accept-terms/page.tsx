"use server";

import { redirect } from "next/navigation";
import { AcceptTermsForm } from "~/components/accept-terms-form";
import { auth } from "~/server/auth";
import { db } from "~/server/db";

export default async function AcceptTermsPage() {
    const session = await auth();

    if (!session?.user?.id) {
        redirect("/login");
    }

    const user = await db.user.findUniqueOrThrow({
        where: { id: session.user.id },
        select: { termsAccepted: true },
    });

    if (user.termsAccepted) {
        redirect("/dashboard");
    }

    return <AcceptTermsForm />;
}