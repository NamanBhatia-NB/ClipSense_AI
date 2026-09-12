"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { signOut } from "next-auth/react";
import { Button } from "~/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "~/components/ui/card";
import { Input } from "~/components/ui/input";
import { deleteAccount } from "~/actions/auth";
import { AlertTriangle } from "lucide-react";

export function DeleteAccountSection() {
    const [confirmText, setConfirmText] = useState("");
    const [isDeleting, setIsDeleting] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const router = useRouter();

    const handleDelete = async () => {
        if (confirmText !== "DELETE") {
            setError("Please type DELETE to confirm");
            return;
        }

        setIsDeleting(true);
        setError(null);

        try {
            const result = await deleteAccount();
            if (result.success) {
                await signOut({ redirect: false });
                router.push("/");
            } else {
                setError(result.error ?? "Failed to delete account");
            }
        } catch (err) {
            setError("An unexpected error occurred");
        } finally {
            setIsDeleting(false);
        }
    };

    return (
        <Card className="border-red-200">
            <CardHeader>
                <CardTitle className="text-red-600">Delete Account</CardTitle>
                <CardDescription>
                    Permanently delete your account and all associated data
                </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
                <div className="rounded-lg bg-red-50 border border-red-200 p-4">
                    <div className="flex gap-3">
                        <AlertTriangle className="h-5 w-5 text-red-600 flex-shrink-0 mt-0.5" />
                        <div className="space-y-2 text-sm">
                            <p className="font-semibold text-red-900">This action cannot be undone. This will permanently:</p>
                            <ul className="list-disc pl-5 space-y-1 text-red-800">
                                <li>Delete your account and profile</li>
                                <li>Remove all uploaded videos from our servers</li>
                                <li>Delete all generated clips permanently</li>
                                <li>Forfeit any remaining credits (non-refundable)</li>
                                <li>Cancel any active subscriptions</li>
                            </ul>
                            <p className="font-semibold text-red-900 mt-3">Account Recovery:</p>
                            <p className="text-red-800">
                                ❌ Your account CANNOT be recovered after deletion. You will need to create a new account if you wish to use the service again.
                            </p>
                        </div>
                    </div>
                </div>

                <div className="space-y-2">
                    <label className="text-sm font-medium">
                        Type <span className="font-mono font-bold">DELETE</span> to confirm
                    </label>
                    <Input
                        value={confirmText}
                        onChange={(e) => setConfirmText(e.target.value)}
                        placeholder="DELETE"
                        className="max-w-xs"
                    />
                </div>

                {error && (
                    <p className="text-sm text-red-500">{error}</p>
                )}

                <Button
                    variant="destructive"
                    onClick={handleDelete}
                    disabled={confirmText !== "DELETE" || isDeleting}
                >
                    {isDeleting ? "Deleting..." : "Delete My Account Permanently"}
                </Button>
            </CardContent>
        </Card>
    );
}