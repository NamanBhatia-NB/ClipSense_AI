"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { signIn } from "next-auth/react";
import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { Loader2 } from "lucide-react";
import { Button } from "~/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "~/components/ui/card";
import { Field, FieldDescription, FieldGroup, FieldLabel } from "~/components/ui/field";
import { Input } from "~/components/ui/input";
import { cn } from "~/lib/utils";
import { loginSchema, type LoginFormValues } from "~/schemas/auth";

export function LoginForm({
    className,
    ...props
}: React.ComponentProps<"div">) {
    const [error, setError] = useState<string | null>(null);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [isGoogleLoading, setIsGoogleLoading] = useState(false);
    const router = useRouter();

    const { register, handleSubmit, formState: { errors } } = useForm<LoginFormValues>({ resolver: zodResolver(loginSchema) });

    const onSubmit = async (data: LoginFormValues) => {
        try {
            setIsSubmitting(true);
            setError(null);

            const signInResult = await signIn("credentials", {
                email: data.email,
                password: data.password,
                redirect: false,
            });

            if (signInResult?.error) {
                setError("Invalid email or password.");
            } else {
                router.push('/dashboard');
            }
        } catch (error) {
            setError("An unexpected error occurred.");
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleGoogleSignIn = async () => {
        try {
            setIsGoogleLoading(true);
            setError(null);
            await signIn("google", { callbackUrl: "/dashboard" });
        } catch (error) {
            setError("An unexpected error occurred with Google sign in.");
            setIsGoogleLoading(false);
        }
    };

    return (
        <div className={cn("flex flex-col gap-6", className)} {...props}>
            <Card>
                <CardHeader>
                    <CardTitle>Login</CardTitle>
                    <CardDescription>
                        Enter your email below to log in for your account
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
                            {error && (
                                <p className="rounded-md bg-red-50 p-3 text-sm text-red-500">{error}</p>
                            )}
                            <Field>
                                <Button type="submit" className="w-full" disabled={isSubmitting || isGoogleLoading}>
                                    {isSubmitting ? (
                                        <>
                                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                            Signing in...
                                        </>
                                    ) : (
                                        "Sign in"
                                    )}
                                </Button>
                                <Button
                                    variant="outline"
                                    type="button"
                                    onClick={handleGoogleSignIn}
                                    disabled={isSubmitting || isGoogleLoading}
                                    className="w-full"
                                >
                                    {isGoogleLoading ? (
                                        <>
                                            <Loader2 className="h-4 w-4 animate-spin" />
                                            Signing in with Google...
                                        </>
                                    ) : (
                                        <>
                                            <Image src="/google.svg" alt="icon" height={20} width={20} />
                                            Login with Google
                                        </>
                                    )}
                                </Button>
                                <FieldDescription className="text-center">
                                    Don&apos;t have an account? {" "}
                                    <Link href="/signup" className="underline underline-offset-4">Sign up</Link>
                                </FieldDescription>
                            </Field>
                        </FieldGroup>
                    </form>
                </CardContent>
            </Card>
        </div>
    )
}
