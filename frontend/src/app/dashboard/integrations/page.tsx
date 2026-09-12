"use client";

import { useState } from "react";
import { signIn } from "next-auth/react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "~/components/ui/card";
import { Button } from "~/components/ui/button";
import { Youtube, Facebook, Instagram, Loader2 } from "lucide-react";

export default function IntegrationsPage() {
  const [isGoogleLoading, setIsGoogleLoading] = useState(false);
  const [isFacebookLoading, setIsFacebookLoading] = useState(false);

  const handleGoogleConnect = async () => {
    try {
      setIsGoogleLoading(true);
      await signIn("google", { callbackUrl: "/dashboard/integrations" });
    } catch {
      setIsGoogleLoading(false);
    }
  };

  const handleFacebookConnect = async () => {
    try {
      setIsFacebookLoading(true);
      await signIn("facebook", { callbackUrl: "/dashboard/integrations" });
    } catch {
      setIsFacebookLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-4xl p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Social Integrations</h1>
        <p className="text-muted-foreground">Connect your accounts to enable one-click publishing and scheduling.</p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {/* YouTube Integration */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Youtube className="text-red-500" /> YouTube
            </CardTitle>
            <CardDescription>Upload Shorts directly to your channel.</CardDescription>
          </CardHeader>
          <CardContent>
            <Button 
              variant="outline" 
              className="w-full"
              disabled={isGoogleLoading || isFacebookLoading}
              onClick={handleGoogleConnect}
            >
              {isGoogleLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Connecting YouTube...
                </>
              ) : (
                "Connect YouTube"
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Meta (Facebook & Instagram) Integration */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Instagram className="text-pink-500" /> <Facebook className="text-blue-600" /> Meta Business
            </CardTitle>
            <CardDescription>Post Reels to your professional Instagram and Facebook Pages.</CardDescription>
          </CardHeader>
          <CardContent>
            <Button 
              variant="outline" 
              className="w-full"
              disabled={isGoogleLoading || isFacebookLoading}
              // Note: Facebook handles both FB Pages and IG Professional accounts
              onClick={handleFacebookConnect}
            >
              {isFacebookLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Connecting Meta...
                </>
              ) : (
                "Connect Meta Accounts"
              )}
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}