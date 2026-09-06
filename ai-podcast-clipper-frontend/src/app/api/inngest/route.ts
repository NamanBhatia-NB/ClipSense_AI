import { serve } from "inngest/next";
import { inngest } from "../../../inngest/client";
import { processVideo, publishSocialVideo } from "~/inngest/functions";

// Create an API that serves zero functions
export const { GET, POST, PUT } = serve({
  client: inngest,
  functions: [
    processVideo, 
    publishSocialVideo // <-- Add the new function here!
  ],
});