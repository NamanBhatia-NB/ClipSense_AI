"use server";

import { Button } from "~/components/ui/button";
import { Badge } from "~/components/ui/badge";
import { Card, CardContent } from "~/components/ui/card";
import Link from "next/link";
import {
  ArrowRight,
  Upload,
  Wand2,
  Smartphone,
  Scissors,
  Zap,
  CheckCircle2,
  Play
} from "lucide-react";
import NavHeader from "~/components/nav-header";
import { auth } from "~/server/auth";
import { db } from "~/server/db";
import Image from "next/image";

export default async function HomePage() {
  const session = await auth();

  let userEmail = null;
  let credits = 0;

  if (session?.user?.id) {
    const user = await db.user.findUniqueOrThrow({
      where: { id: session.user.id },
      select: { credits: true, email: true },
    });

    if (user) {
      userEmail = user.email;
      credits = user.credits;
    }
  }

  return (
    <div className="flex flex-col min-h-screen">
      <NavHeader credits={credits} user={userEmail} />
      {/* 1. HERO SECTION WITH GRID BACKGROUND */}
      <section className="relative w-full pt-32 pb-24 lg:pt-48 lg:pb-32 overflow-hidden">
        {/* Background Pattern */}
        <div className="absolute inset-0 -z-10 h-full w-full bg-white bg-[linear-gradient(to_right,#f0f0f0_1px,transparent_1px),linear-gradient(to_bottom,#f0f0f0_1px,transparent_1px)] bg-[size:6rem_4rem]">
          <div className="absolute bottom-0 left-0 right-0 top-0 bg-[radial-gradient(circle_500px_at_50%_200px,#C9EBFF,transparent)]"></div>
        </div>

        <div className="container px-4 md:px-6 mx-auto text-center">
          <Badge variant="secondary" className="mb-6 px-4 py-2 text-sm border-primary/20 bg-primary/5 text-primary rounded-full animate-fade-in-up">
            ✨ AI-Powered Podcast Editor
          </Badge>

          <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold tracking-tight text-slate-900 mb-6 max-w-4xl mx-auto leading-tight">
            Turn Long Podcasts into <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-violet-600">
              Viral Shorts
            </span> in Seconds
          </h1>

          <p className="text-lg md:text-xl text-slate-600 mb-10 max-w-2xl mx-auto leading-relaxed">
            Stop spending hours editing. <br /> Our AI automatically identifies highlights, crops active speakers, adds captions, and creates vertical clips for YouTube Shorts, Instagram Reels & TikTok.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <Button size="default" className="h-14 px-8 text-lg rounded-full shadow-lg shadow-blue-600/20 hover:shadow-blue-600/30 transition-all w-full sm:w-auto" asChild>
              <Link href="/dashboard">
                Try for Free <ArrowRight className="ml-2 h-5 w-5" />
              </Link>
            </Button>
            <Button variant="outline" size="default" className="h-14 px-8 text-lg rounded-full w-full sm:w-auto" asChild>
              <Link href="#demo">
                <Play className="mr-2 h-5 w-5 fill-current" /> Watch Demo
              </Link>
            </Button>
          </div>

          <div className="mt-8 flex items-center justify-center gap-4 text-sm text-slate-500">
            <div className="flex items-center gap-1">
              <CheckCircle2 className="h-4 w-4 text-green-500" /> No credit card required
            </div>
            <div className="flex items-center gap-1">
              <CheckCircle2 className="h-4 w-4 text-green-500" /> 1 clip = 1 credit
            </div>
          </div>
        </div>
      </section>

      {/* 2. SOCIAL PROOF / LOGO TICKER */}
      <section className="py-10 border-y bg-slate-50/50">
        <div className="container px-4 md:px-6 mx-auto">
          <p className="text-center text-sm font-semibold text-slate-500 uppercase tracking-wider mb-8">
            Trusted by creators from
          </p>
          <div className="flex flex-wrap justify-center gap-8 md:gap-16 opacity-40 grayscale hover:grayscale-0 transition-all duration-500">
            {/* Placeholder logos - replace with SVGs later */}
            <span className="text-xl font-bold">YouTube</span>
            <span className="text-xl font-bold">Spotify</span>
            <span className="text-xl font-bold">TikTok</span>
            <span className="text-xl font-bold">Instagram</span>
            <span className="text-xl font-bold">LinkedIn</span>
          </div>
        </div>
      </section>

      {/* 3. VALUE PROP: BENTO GRID STYLE */}
      <section className="py-24 bg-white" id="features">
        <div className="container px-4 md:px-6 mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-5xl font-bold tracking-tight mb-4">
              Everything you need to go viral
            </h2>
            <p className="text-slate-600 text-lg max-w-2xl mx-auto">
              We don&apos;t just cut video. We engineer content for retention using advanced AI models.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 auto-rows-[300px]">
            {/* Feature 1: Large */}
            <Card className="md:col-span-2 bg-slate-50 border-slate-200 overflow-hidden relative group">
              <CardContent className="p-8 h-full flex flex-col justify-between z-10 relative">
                <div>
                  <div className="h-10 w-10 bg-blue-100 rounded-lg flex items-center justify-center mb-4 text-blue-600">
                    <Wand2 className="h-6 w-6" />
                  </div>
                  <h3 className="text-2xl font-bold mb-2">AI Magic Curation</h3>
                  <p className="text-slate-600 max-w-md">Our algorithms analyze engagement signals to pick the funniest, smartest, and most viral moments from your hour-long episodes.</p>
                </div>
                {/* Mockup visual */}
                <div className="absolute right-0 bottom-0 w-1/2 h-4/5 bg-gradient-to-tl from-blue-500/10 to-transparent rounded-tl-3xl border-t border-l border-slate-200/50 translate-x-4 translate-y-4"></div>
              </CardContent>
            </Card>

            {/* Feature 2: Tall Vertical */}
            <Card className="md:row-span-2 bg-slate-900 text-white overflow-hidden relative">
              <CardContent className="p-8 h-full flex flex-col z-10 relative">
                <div className="h-10 w-10 bg-white/10 rounded-lg flex items-center justify-center mb-4 text-white">
                  <Smartphone className="h-6 w-6" />
                </div>
                <h3 className="text-2xl font-bold mb-2">9:16 Vertical Perfection</h3>
                <p className="text-slate-300 mb-8">AI automatically tracks faces and switches cameras to keep the active speaker in focus.</p>

                {/* Visual representation of a phone screen */}
                <div className="flex-1 w-full bg-slate-800 rounded-t-2xl border border-slate-700 mx-auto max-w-[200px] relative overflow-hidden shadow-2xl">
                  <div className="absolute inset-0 flex items-center justify-center text-slate-600 text-xs">
                    [Generated Clip Preview]
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Feature 3 */}
            <Card className="bg-slate-50 border-slate-200">
              <CardContent className="p-8 flex flex-col justify-center h-full">
                <div className="h-10 w-10 bg-purple-100 rounded-lg flex items-center justify-center mb-4 text-purple-600">
                  <Scissors className="h-6 w-6" />
                </div>
                <h3 className="text-xl font-bold mb-2">Auto-Cropping</h3>
                <p className="text-slate-600">We detect faces and crop dynamically. No more manual keyframing.</p>
              </CardContent>
            </Card>

            {/* Feature 4 */}
            <Card className="bg-slate-50 border-slate-200">
              <CardContent className="p-8 flex flex-col justify-center h-full">
                <div className="h-10 w-10 bg-orange-100 rounded-lg flex items-center justify-center mb-4 text-orange-600">
                  <Zap className="h-6 w-6" />
                </div>
                <h3 className="text-xl font-bold mb-2">Lightning Fast</h3>
                <p className="text-slate-600">Processing happens in the cloud. Get 10 clips in the time it takes to grab coffee.</p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* 4. HOW IT WORKS (Step by Step) */}
      <section className="py-24 bg-slate-50">
        <div className="container px-4 md:px-6 mx-auto">
          <h2 className="text-3xl font-bold text-center mb-16">From Upload to Viral in 3 Steps</h2>

          <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
            {[
              { title: "Upload Video", desc: "Paste a YouTube link or upload your raw file.", icon: Upload },
              { title: "AI Processing", desc: "We analyze, clip, caption, and reframe automatically.", icon: Wand2 },
              { title: "Download & Post", desc: "Get high-res MP4s ready for TikTok and Reels.", icon: ArrowRight },
            ].map((step, i) => (
              <div key={i} className="flex flex-col items-center text-center group">
                <div className="w-16 h-16 bg-white rounded-2xl shadow-sm border border-slate-200 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                  <step.icon className="h-8 w-8 text-slate-900" />
                </div>
                <h3 className="text-xl font-bold mb-2">{step.title}</h3>
                <p className="text-slate-600">{step.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* 5. PRICING TEASER / CTA */}
      <section className="py-24 border-t relative overflow-hidden">
        <div className="absolute inset-0 -z-10 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-blue-100/50 via-transparent to-transparent"></div>

        <div className="container px-4 md:px-6 mx-auto text-center max-w-3xl">
          <h2 className="text-4xl font-bold tracking-tight mb-6">Stop paying monthly subscriptions for unused minutes.</h2>
          <p className="text-xl text-slate-600 mb-8">
            Pay only for the clips you generate. Credits never expire.
          </p>

          <div className="flex flex-col sm:flex-row justify-center gap-4">
            <Button size="default" className="h-14 px-8 rounded-full text-lg" asChild>
              <Link href="/dashboard">
                Start Creating for Free
              </Link>
            </Button>
            <Button variant="secondary" size="lg" className="h-14 rounded-full" asChild>
              <Link href="/dashboard/billing">
                View Pricing
              </Link>
            </Button>
          </div>
          <p className="mt-6 text-sm text-slate-500">
            Sign up now and get your first five clips for free on the house.
          </p>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="py-3 bg-white border-t">
        <div className="container px-4 md:px-6 mx-auto flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="flex justify-center items-center gap-1 font-bold text-lg">
            <div className="h-6 w-6 rounded-lg flex items-center justify-center">
              <Image src="/logo.png" alt="" width={20} height={20} />
            </div>
            AI Podcast/Clipper
          </div>
          <p className="text-slate-500 text-sm">
            © {new Date().getFullYear()} AI Podcast/Clipper Inc. All rights reserved.
          </p>
          <div className="flex gap-6 text-sm font-medium text-slate-600">
            <Link href="/privacy" className="hover:text-slate-900">Privacy</Link>
            <Link href="/terms" className="hover:text-slate-900">Terms</Link>
          </div>
        </div>
      </footer>

    </div>
  );
}