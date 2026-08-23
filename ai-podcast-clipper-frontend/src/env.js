import { createEnv } from "@t3-oss/env-nextjs";
import { z } from "zod";

export const env = createEnv({
  /**
   * Specify your server-side environment variables schema here. This way you can ensure the app
   * isn't built with invalid env vars.
   */
  server: {
    AUTH_SECRET:
      process.env.NODE_ENV === "production"
        ? z.string()
        : z.string().optional(),
    DATABASE_URL: z.string().url(),
    NODE_ENV: z
      .enum(["development", "test", "production"])
      .default("development"),
    AWS_ACCESS_KEY_ID: z.string(),
    AWS_SECRET_ACCESS_KEY: z.string(),
    AWS_REGION: z.string(),
    S3_BUCKET_NAME: z.string(),
    PROCESS_VIDEO_ENDPOINT: z.string(),
    PROCESS_VIDEO_ENDPOINT_AUTH: z.string(),
    BASE_URL: z.string(),
    AUTH_GOOGLE_ID:z.string(),
    AUTH_GOOGLE_SECRET:z.string(),
    POLAR_ACCESS_TOKEN:z.string(),
    POLAR_WEBHOOK_SECRET:z.string(),
    POLAR_PRODUCT_SMALL_ID:z.string(),
    POLAR_PRODUCT_MEDIUM_ID:z.string(),
    POLAR_PRODUCT_LARGE_ID:z.string(),
    AUTH_FACEBOOK_ID:z.string(),
    AUTH_FACEBOOK_SECRET:z.string(),
    AUTH_TWITTER_ID:z.string(),
    AUTH_TWITTER_SECRET:z.string(),
  },

  /**
   * Specify your client-side environment variables schema here. This way you can ensure the app
   * isn't built with invalid env vars. To expose them to the client, prefix them with
   * `NEXT_PUBLIC_`.
   */
  client: {
    // NEXT_PUBLIC_CLIENTVAR: z.string(),
  },

  /**
   * You can't destruct `process.env` as a regular object in the Next.js edge runtimes (e.g.
   * middlewares) or client-side so we need to destruct manually.
   */
  runtimeEnv: {
    AUTH_SECRET: process.env.AUTH_SECRET,
    DATABASE_URL: process.env.DATABASE_URL,
    NODE_ENV: process.env.NODE_ENV,
    AWS_ACCESS_KEY_ID: process.env.AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY: process.env.AWS_SECRET_ACCESS_KEY,
    AWS_REGION: process.env.AWS_REGION,
    S3_BUCKET_NAME: process.env.S3_BUCKET_NAME,
    PROCESS_VIDEO_ENDPOINT: process.env.PROCESS_VIDEO_ENDPOINT,
    PROCESS_VIDEO_ENDPOINT_AUTH: process.env.PROCESS_VIDEO_ENDPOINT_AUTH,
    BASE_URL: process.env.BASE_URL,
    AUTH_GOOGLE_ID:process.env.AUTH_GOOGLE_ID,
    AUTH_GOOGLE_SECRET:process.env.AUTH_GOOGLE_SECRET,
    AUTH_FACEBOOK_ID:process.env.AUTH_FACEBOOK_ID,
    AUTH_FACEBOOK_SECRET:process.env.AUTH_FACEBOOK_SECRET,
    POLAR_ACCESS_TOKEN:process.env.POLAR_ACCESS_TOKEN,
    POLAR_WEBHOOK_SECRET:process.env.POLAR_WEBHOOK_SECRET,
    POLAR_PRODUCT_SMALL_ID:process.env.POLAR_PRODUCT_SMALL_ID,
    POLAR_PRODUCT_MEDIUM_ID:process.env.POLAR_PRODUCT_MEDIUM_ID,
    POLAR_PRODUCT_LARGE_ID:process.env.POLAR_PRODUCT_LARGE_ID,
    AUTH_TWITTER_ID:process.env.AUTH_TWITTER_ID,
    AUTH_TWITTER_SECRET:process.env.AUTH_TWITTER_SECRET,
  },
  /**
   * Run `build` or `dev` with `SKIP_ENV_VALIDATION` to skip env validation. This is especially
   * useful for Docker builds.
   */
  skipValidation: !!process.env.SKIP_ENV_VALIDATION,
  /**
   * Makes it so that empty strings are treated as undefined. `SOME_VAR: z.string()` and
   * `SOME_VAR=''` will throw an error.
   */
  emptyStringAsUndefined: true,
});
