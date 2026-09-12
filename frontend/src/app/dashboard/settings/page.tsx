"use server";

import { redirect } from "next/navigation";
import { DeleteAccountSection } from "~/components/delete-account-section";
import { auth } from "~/server/auth";

export default async function SettingsPage() {
    const session = await auth();

    if (!session?.user?.id) {
        redirect("/login");
    }

    return (
        <div className="container mx-auto max-w-4xl px-4 py-8">
            <h1 className="text-3xl font-bold mb-8">Account Settings</h1>
            <DeleteAccountSection />
        </div>
    );
}