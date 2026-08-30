"use client";

import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuLabel,
    DropdownMenuSeparator,
    DropdownMenuTrigger
} from "./ui/dropdown-menu";
import { Avatar, AvatarFallback } from "./ui/avatar";
import { signOut } from "next-auth/react";
import Link from "next/link";
import { ArrowRight, Menu } from "lucide-react";
import Image from "next/image";

type NavHeaderProps = {
    user?: string | null;
    credits?: number;
};

const NavHeader = ({ user, credits = 0 }: NavHeaderProps) => {
    return (
        <header className="bg-background/80 backdrop-blur-md sticky top-0 z-50 border-b w-full">
            <div className="container mx-auto flex h-16 items-center justify-between px-4 md:px-6">
                {/* Logo */}
                <Link href="/" className="flex items-center gap-2 transition-opacity hover:opacity-80">
                    <div className="h-8 w-8 rounded-lg flex items-center justify-center">
                        <Image src="/logo.png" alt=""  width={20} height={20}/>
                    </div>
                    <div className="font-sans text-xl font-medium tracking-tight">
                        <span className="text-foreground">AI Podcast</span>
                        <span className="font-light text-gray-500">/</span>
                        <span className="text-foreground font-light">Clipper</span>
                    </div>
                </Link>

                {/* Actions */}
                <div className="flex items-center gap-4">
                    {/* Check if user (email string) exists */}
                    {user ? (
                        /* LOGGED IN STATE */
                        <div className="flex items-center gap-2 sm:gap-4">
                            
                            {/* CHANGED: Removed 'hidden sm:flex' so this shows on all screens */}
                            <div className="hidden sm:flex items-center gap-2">
                                <Badge
                                    variant="secondary"
                                    className="h-8 px-2 sm:px-3 py-1.5 text-[10px] sm:text-xs font-medium border border-border whitespace-nowrap"
                                >
                                    {credits} credits
                                </Badge>
                                <Button
                                    variant="outline"
                                    size="sm"
                                    asChild
                                    className="h-8 text-xs font-medium px-2 sm:px-3"
                                >
                                    <Link href="/dashboard/billing">
                                        {/* Optional: Show "Buy" on tiny screens, "Buy more" on larger */}
                                        <span className="sm:hidden">Buy</span>
                                        <span className="hidden sm:inline">Buy more</span>
                                    </Link>
                                </Button>
                            </div>

                            <DropdownMenu>
                                <DropdownMenuTrigger asChild>
                                    <Button
                                        variant="ghost"
                                        className="relative h-9 w-9 rounded-full ring-2 ring-transparent hover:ring-primary/20 transition-all"
                                    >
                                        <Avatar className="h-9 w-9 border">
                                            <AvatarFallback className="bg-primary/10 text-primary font-semibold">
                                                {user.charAt(0).toUpperCase()}
                                            </AvatarFallback>
                                        </Avatar>
                                    </Button>
                                </DropdownMenuTrigger>
                                <DropdownMenuContent align="end" className="w-56">
                                    <DropdownMenuLabel>
                                        <p className="font-normal text-muted-foreground text-xs truncate">
                                            {user}
                                        </p>
                                    </DropdownMenuLabel>
                                    <DropdownMenuSeparator />
                                    <DropdownMenuItem asChild className="sm:hidden">
                                        <Link href="/dashboard/billing" className="cursor-pointer font-medium">Buy Credits</Link>
                                    </DropdownMenuItem>
                                    <DropdownMenuSeparator className="sm:hidden" />
                                    <DropdownMenuItem asChild>
                                        <Link href="/dashboard" className="cursor-pointer">Dashboard</Link>
                                    </DropdownMenuItem>
                                    <DropdownMenuItem asChild>
                                        <Link href="/dashboard/billing" className="cursor-pointer">Billing & Credits</Link>
                                    </DropdownMenuItem>
                                    <DropdownMenuItem asChild>
                                        <Link href="/dashboard/settings" className="cursor-pointer">Settings</Link>
                                    </DropdownMenuItem>
                                    <DropdownMenuSeparator />
                                    <DropdownMenuItem
                                        onClick={() => signOut({ callbackUrl: '/' })}
                                        className="text-destructive focus:text-destructive cursor-pointer"
                                    >
                                        Sign out
                                    </DropdownMenuItem>
                                </DropdownMenuContent>
                            </DropdownMenu>
                        </div>
                    ) : (
                        /* LOGGED OUT STATE */
                        <>
                            {/* Desktop View (Hidden on mobile) */}
                            <div className="hidden md:flex items-center gap-3">
                                <Button variant="ghost" size="sm" asChild className="text-muted-foreground hover:text-foreground">
                                    <Link href="/login">Log in</Link>
                                </Button>
                                <Button size="sm" asChild className="group">
                                    <Link href="/signup">
                                        Get Started
                                        <ArrowRight className="ml-2 h-4 w-4 group-hover:translate-x-1 transition-transform" />
                                    </Link>
                                </Button>
                            </div>

                            {/* Mobile View (Hamburger Menu) */}
                            <div className="md:hidden">
                                <DropdownMenu>
                                    <DropdownMenuTrigger asChild>
                                        <Button variant="ghost" size="icon" suppressHydrationWarning>
                                            <Menu className="h-5 w-5" />
                                            <span className="sr-only">Toggle menu</span>
                                        </Button>
                                    </DropdownMenuTrigger>
                                    <DropdownMenuContent align="end" className="w-48">
                                        <DropdownMenuItem asChild>
                                            <Link href="/login" className="flex w-full cursor-pointer items-center justify-between">
                                                Log in
                                            </Link>
                                        </DropdownMenuItem>
                                        <DropdownMenuSeparator />
                                        <DropdownMenuItem asChild>
                                            <Link href="/signup" className="flex w-full cursor-pointer items-center justify-between font-bold text-primary">
                                                Get Started
                                                <ArrowRight className="h-4 w-4" />
                                            </Link>
                                        </DropdownMenuItem>
                                    </DropdownMenuContent>
                                </DropdownMenu>
                            </div>
                        </>
                    )}
                </div>
            </div>
        </header>
    );
};

export default NavHeader;