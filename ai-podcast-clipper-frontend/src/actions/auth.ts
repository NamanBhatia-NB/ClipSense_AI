"use server";

import { hashPassword } from "~/lib/auth";
import { signupSchema, type SignupFormValues } from "~/schemas/auth";
import { db } from "~/server/db";
import { auth } from "~/server/auth";

type SignupResult = {
  success: boolean;
  error?: string;
};

export async function signup(data: SignupFormValues): Promise<SignupResult> {
  const validationResult = signupSchema.safeParse(data);
  if (!validationResult.success) {
    return {
      success: false,
      error: validationResult.error.issues[0]?.message ?? "Invalid input",
    };
  }

  const { email, password } = validationResult.data;

  try {
    const existingUser = await db.user.findUnique({
      where: { email },
    });

    if (existingUser) {
      return {
        success: false,
        error: "Email already in use",
      };
    }

    const hashedPassword = await hashPassword(password);

    await db.user.create({
      data: {
        email,
        password: hashedPassword,
        termsAccepted: true,
        termsAcceptedAt: new Date(),
      },
    });

    return {
      success: true,
    };
  } catch (error) {
    console.error("Error during signup:", error);
    return {
      success: false,
      error: "An error occurred during signup",
    };
  }
}

export async function acceptTerms(): Promise<SignupResult> {
  const session = await auth();
  
  if (!session?.user?.id) {
    return { success: false, error: "Not authenticated" };
  }

  try {
    await db.user.update({
      where: { id: session.user.id },
      data: {
        termsAccepted: true,
        termsAcceptedAt: new Date(),
      },
    });

    return { success: true };
  } catch (error) {
    return { success: false, error: "Failed to accept terms" };
  }
}

export async function deleteAccount(): Promise<SignupResult> {
  const session = await auth();
  
  if (!session?.user?.id) {
    return { success: false, error: "Not authenticated" };
  }

  try {
    // Delete user (cascade will delete all related data)
    await db.user.delete({
      where: { id: session.user.id },
    });

    return { success: true };
  } catch (error) {
    return { success: false, error: "Failed to delete account" };
  }
}
