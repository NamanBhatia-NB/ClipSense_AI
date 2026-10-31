import { createEnv } from "@t3-oss/env-nextjs";
import { z } from "zod";

export const env = createEnv({
  /**
   * Specify your server-side environment variables schema here.
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
    BASE_URL: z.string().default("http://localhost:3000"),
    AUTH_GOOGLE_ID: z.string().optional(),
    AUTH_GOOGLE_SECRET: z.string().optional(),
    POLAR_ACCESS_TOKEN: z.string().optional(),
    POLAR_WEBHOOK_SECRET: z.string().optional(),
    POLAR_PRODUCT_SMALL_ID: z.string().optional(),
    POLAR_PRODUCT_MEDIUM_ID: z.string().optional(),
    POLAR_PRODUCT_LARGE_ID: z.string().optional(),
  },

  /**
   * Specify your client-side environment variables schema here.
   */
  client: {
    NEXT_PUBLIC_CLIPSENSE_API_URL: z.string().default("http://localhost:8000"),
  },

  /**
   * Runtime env mappings
   */
  runtimeEnv: {
    NEXT_PUBLIC_CLIPSENSE_API_URL: process.env.NEXT_PUBLIC_CLIPSENSE_API_URL,
    AUTH_SECRET: process.env.AUTH_SECRET,
    DATABASE_URL: process.env.DATABASE_URL,
    NODE_ENV: process.env.NODE_ENV,
    BASE_URL: process.env.BASE_URL,
    AUTH_GOOGLE_ID: process.env.AUTH_GOOGLE_ID,
    AUTH_GOOGLE_SECRET: process.env.AUTH_GOOGLE_SECRET,
    POLAR_ACCESS_TOKEN: process.env.POLAR_ACCESS_TOKEN,
    POLAR_WEBHOOK_SECRET: process.env.POLAR_WEBHOOK_SECRET,
    POLAR_PRODUCT_SMALL_ID: process.env.POLAR_PRODUCT_SMALL_ID,
    POLAR_PRODUCT_MEDIUM_ID: process.env.POLAR_PRODUCT_MEDIUM_ID,
    POLAR_PRODUCT_LARGE_ID: process.env.POLAR_PRODUCT_LARGE_ID,
  },

  /**
   * Run `build` or `dev` with `SKIP_ENV_VALIDATION` to skip env validation.
   */
  skipValidation: !!process.env.SKIP_ENV_VALIDATION,
  emptyStringAsUndefined: true,
});
