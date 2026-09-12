"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { Button } from "~/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "~/components/ui/card";
import { Field, FieldDescription, FieldGroup, FieldLabel } from "~/components/ui/field";
import { Input } from "~/components/ui/input";
import { cn } from "~/lib/utils";
import { signupSchema, type SignupFormValues } from "~/schemas/auth";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { signup } from "~/actions/auth";
import { signIn } from "next-auth/react";
import Link from "next/link";
import Image from "next/image";

import { Loader2 } from "lucide-react";

export function SignupForm({
    className,
    ...props
}: React.ComponentProps<"div">) {
    const [error, setError] = useState<string | null>(null);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [isGoogleLoading, setIsGoogleLoading] = useState(false);
    const router = useRouter();

    const { register, handleSubmit, formState: { errors } } = useForm<SignupFormValues>({ resolver: zodResolver(signupSchema) });

    const onSubmit = async (data: SignupFormValues) => {
        try {
            setIsSubmitting(true);
            setError(null);

            const result = await signup(data);
            if (!result.success) {
                setError(result.error ?? "An error occurred during signup.");
                return;
            }

            const signUpResult = await signIn("credentials", {
                email: data.email,
                password: data.password,
                redirect: false,
            });

            if (signUpResult?.error) {
                setError("Account created but couldn't sign in automatically. Please try again.",);
            } else {
                router.push('/dashboard');
            }
        } catch (error) {
            setError("An unexpected error occurred.");
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleGoogleSignUp = async () => {
        try {
            setIsGoogleLoading(true);
            setError(null);
            await signIn("google", { callbackUrl: "/dashboard" });
        } catch (error) {
            setError("An unexpected error occurred with Google sign up.");
            setIsGoogleLoading(false);
        }
    };

    return (
        <div className={cn("flex flex-col gap-6", className)} {...props}>
            <Card>
                <CardHeader>
                    <CardTitle>Sign up</CardTitle>
                    <CardDescription>
                        Enter your email below to sign up for your account
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <form onSubmit={handleSubmit(onSubmit)}>
                        <FieldGroup>
                            <Field>
                                <FieldLabel htmlFor="email">Email</FieldLabel>
                                <Input
                                    id="email"
                                    type="email"
                                    placeholder="m@example.com"
                                    required
                                    {...register("email")}
                                />
                                {errors.email && (
                                    <p className="text-sm text-red-500">{errors.email.message}</p>
                                )}
                            </Field>
                            <Field>
                                <div className="flex items-center">
                                    <FieldLabel htmlFor="password">Password</FieldLabel>
                                </div>
                                <Input id="password" type="password" required {...register("password")} />
                                {errors.password && (
                                    <p className="text-sm text-red-500">{errors.password.message}</p>
                                )}
                            </Field>
                            <Field>
                                <div className="flex items-start gap-2">
                                    <input
                                        type="checkbox"
                                        id="termsAccepted"
                                        className="mt-1"
                                        {...register("termsAccepted")}
                                    />
                                    <FieldLabel htmlFor="termsAccepted" className="text-sm font-normal inline">
                                        I agree to the {" "}
                                        <Link href="/terms" target="_blank" className="underline">
                                             Terms and Conditions
                                        </Link>
                                        {" "}and{" "}
                                        <Link href="/privacy" target="_blank" className="underline">
                                             Privacy Policy
                                        </Link>
                                    </FieldLabel>
                                </div>
                                {errors.termsAccepted && (
                                    <p className="text-sm text-red-500">{errors.termsAccepted.message}</p>
                                )}
                            </Field>
                            {error && (
                                <p className="rounded-md bg-red-50 p-3 text-sm text-red-500">{error}</p>
                            )}
                            <Field>
                                <Button type="submit" className="w-full" disabled={isSubmitting || isGoogleLoading}>
                                    {isSubmitting ? (
                                        <>
                                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                            Signing up...
                                        </>
                                    ) : (
                                        "Sign up"
                                    )}
                                </Button>
                                <Button
                                    variant="outline"
                                    type="button"
                                    onClick={handleGoogleSignUp}
                                    disabled={isSubmitting || isGoogleLoading}
                                    className="w-full"
                                >
                                    {isGoogleLoading ? (
                                        <>
                                            <Loader2 className="h-4 w-4 animate-spin" />
                                            Signing up with Google...
                                        </>
                                    ) : (
                                        <>
                                            <Image src="/google.svg" alt="icon" height={20} width={20} />
                                            Sign up with Google
                                        </>
                                    )}
                                </Button>
                                <FieldDescription className="text-center">
                                    Already have an account? {" "}
                                    <Link href="/login" className="underline underline-offset-4">Sign in</Link>
                                </FieldDescription>
                            </Field>
                        </FieldGroup>
                    </form>
                </CardContent>
            </Card>
        </div>
    )
}
