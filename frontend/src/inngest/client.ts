import { Inngest, EventSchemas } from "inngest";

// Define the shape of your existing video processing event
type ProcessVideoEvent = {
  data: {
    uploadedFileId: string;
    userId: string;
    clipSettings: {
      mode: "auto" | "manual";
      requestedClips?: number;
      userPrompt?: string;
    };
  };
};

// Define the shape of your NEW social publishing event
type PublishSocialVideoEvent = {
  data: {
    userId: string;
    clipId: string;
    platforms: {
      youtube: boolean;
      instagram: boolean;
      facebook: boolean;
      // x: boolean;
    };
    ytTitle: string;
    ytDesc: string;
    instaCaption: string;
    fbCaption: string;
    // xCaption: string;
    scheduledTime: string | null;
  };
};

// Create a client and inject the schemas
export const inngest = new Inngest({
  id: "ai-podcast-clipper-frontend",
  schemas: new EventSchemas().fromRecord<{
    "process-video-events": ProcessVideoEvent;
    "publish-social-video": PublishSocialVideoEvent;
  }>(),
});