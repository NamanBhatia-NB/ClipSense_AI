"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Button } from "~/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "~/components/ui/card";
import { acceptTerms } from "~/actions/auth";

export function AcceptTermsForm() {
    const [accepted, setAccepted] = useState(false);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const router = useRouter();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        
        if (!accepted) {
            setError("You must accept the terms to continue");
            return;
        }

        setIsSubmitting(true);
        setError(null);

        try {
            const result = await acceptTerms();
            if (result.success) {
                router.push("/dashboard");
            } else {
                setError(result.error ?? "Failed to accept terms");
            }
        } catch (err) {
            setError("An unexpected error occurred");
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <div className="flex min-h-svh w-full items-center justify-center p-6 md:p-10">
            <Card className="w-full max-w-2xl">
                <CardHeader>
                    <CardTitle>Accept Terms and Conditions</CardTitle>
                    <CardDescription>
                        Please review and accept our terms to continue
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <form onSubmit={handleSubmit} className="space-y-6">
                        <div className="max-h-96 overflow-y-auto rounded-lg border p-4 bg-slate-50">
                            <h3 className="font-semibold mb-2">Key Points:</h3>
                            <ul className="list-disc pl-6 space-y-2 text-sm text-slate-600">
                                <li>AI-generated content may contain errors or inaccuracies</li>
                                <li>You must review all clips and captions before publishing</li>
                                <li>You are responsible for content you upload</li>
                                <li>Credits are non-refundable and never expire</li>
                                <li>We are not liable for AI-generated content errors</li>
                            </ul>
                            <div className="mt-4 space-x-4">
                                <Link href="/terms" target="_blank" className="text-sm text-blue-600 underline">
                                    Read Full Terms
                                </Link>
                                <Link href="/privacy" target="_blank" className="text-sm text-blue-600 underline">
                                    Read Privacy Policy
                                </Link>
                            </div>
                        </div>

                        <div className="flex items-start gap-2">
                            <input
                                type="checkbox"
                                id="accept"
                                checked={accepted}
                                onChange={(e) => setAccepted(e.target.checked)}
                                className="mt-1"
                            />
                            <label htmlFor="accept" className="text-sm">
                                I have read and agree to the Terms and Conditions and Privacy Policy
                            </label>
                        </div>

                        {error && (
                            <p className="text-sm text-red-500">{error}</p>
                        )}

                        <Button type="submit" className="w-full" disabled={!accepted || isSubmitting}>
                            {isSubmitting ? "Processing..." : "Accept and Continue"}
                        </Button>
                    </form>
                </CardContent>
            </Card>
        </div>
    );
}