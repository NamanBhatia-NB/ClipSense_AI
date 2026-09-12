"use server";

import { env } from "~/env";
import { Polar } from "@polar-sh/sdk";
import { db } from "~/server/db";
import { auth } from "~/server/auth";
import { redirect } from "next/navigation";

// Initialize Polar
const polar = new Polar({
  accessToken: process.env.POLAR_ACCESS_TOKEN,
  server: process.env.NODE_ENV === "development" ? "sandbox" : "production",
  // server: "production",
});

export type PriceId = "small" | "medium" | "large";

// Map your internal keys to Polar Product IDs
const PRODUCT_IDS: Record<PriceId, string> = {
  small: process.env.POLAR_PRODUCT_SMALL_ID!,
  medium: process.env.POLAR_PRODUCT_MEDIUM_ID!,
  large: process.env.POLAR_PRODUCT_LARGE_ID!,
};

export async function createCheckoutSession(priceId: PriceId) {
  const serverSession = await auth();

  // 1. Fetch user to get ID and Email
  const user = await db.user.findUniqueOrThrow({
    where: {
      id: serverSession?.user.id,
    },
    select: { 
      id: true, 
      email: true 
    },
  });

  // 2. Create Polar Checkout
  const checkout = await polar.checkouts.create({
    products: [PRODUCT_IDS[priceId]], // Pass the Product ID here
    successUrl: `${env.BASE_URL}/dashboard?success=true`,
    customerEmail: user.email, // Pre-fills the email for the user
    metadata: {
      userId: user.id, // CRITICAL: This lets you identify the user in your webhook
    },
  });

  // 3. Redirect to the Polar checkout URL
  if (!checkout.url) {
    throw new Error("Failed to create checkout session URL.");
  }

  redirect(checkout.url);
}