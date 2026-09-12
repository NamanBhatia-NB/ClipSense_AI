import { PrismaAdapter } from "@auth/prisma-adapter";
import { type DefaultSession, type NextAuthConfig } from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";
import Google from "next-auth/providers/google";
import Facebook from "next-auth/providers/facebook";
// import Twitter from "next-auth/providers/twitter";
import { env } from "~/env";
import { comparePasswords } from "~/lib/auth";
import { db } from "~/server/db";

/**
 * Module augmentation for `next-auth` types. Allows us to add custom properties to the `session`
 * object and keep type safety.
 *
 * @see https://next-auth.js.org/getting-started/typescript#module-augmentation
 */
declare module "next-auth" {
  interface Session extends DefaultSession {
    user: {
      id: string;
      // ...other properties
      // role: UserRole;
    } & DefaultSession["user"];
  }

  // interface User {
  //   // ...other properties
  //   // role: UserRole;
  // }
}

/**
 * Options for NextAuth.js used to configure adapters, providers, callbacks, etc.
 *
 * @see https://next-auth.js.org/configuration/options
 */
export const authConfig = {
  providers: [
    CredentialsProvider({
      name: "credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          return null;
        }

        const email = credentials.email as string;
        const password = credentials.password as string;

        const user = await db.user.findUnique({
          where: { email },
        });

        if (!user) {
          return null;
        }

        const passwordMatch = await comparePasswords(password, user.password!);
        if (!passwordMatch) return null;

        return user;
      },
    }),
    Google({
      clientId: env.AUTH_GOOGLE_ID,
      clientSecret: env.AUTH_GOOGLE_SECRET,
      allowDangerousEmailAccountLinking: true,
      authorization: {
        params: {
          prompt: "consent",
          access_type: "offline", // CRITICAL: Forces Google to give a refresh_token
          response_type: "code",
          // CRITICAL: Adds YouTube upload permission
          scope: "openid email profile https://www.googleapis.com/auth/youtube.upload",
        },
      },
    }),
    Facebook({
      clientId: env.AUTH_FACEBOOK_ID, // ADD TO ENV
      clientSecret: env.AUTH_FACEBOOK_SECRET, // ADD TO ENV
      allowDangerousEmailAccountLinking: true,
      authorization: {
        params: {
          // CRITICAL: Adds Instagram and Facebook Page posting permissions
          scope: "email,public_profile,pages_show_list,pages_read_engagement,pages_manage_posts,instagram_basic,instagram_content_publish,business_management",
        },
      },
    }),
    /*
    Twitter({
      clientId: env.AUTH_TWITTER_ID,
      clientSecret: env.AUTH_TWITTER_SECRET,
      authorization: {
        // EXPLICITLY define the URL so NextAuth doesn't lose it
        url: "https://twitter.com/i/oauth2/authorize",
        params: {
          // 'offline.access' is required to get a refresh_token
          scope: "tweet.read tweet.write users.read media.write offline.access",
        },
      },
    }),
    */
  ],
  events: {
    createUser: async ({ user }) => {
      if (user.email) {
        await db.user.update({
          where: { id: user.id },
          data: {
            termsAccepted: false,
          },
        });
      }
    },
  },
  session: { strategy: "jwt" },
  adapter: PrismaAdapter(db),
  callbacks: {
    session: ({ session, token }) => ({
      ...session,
      user: {
        ...session.user,
        id: token.sub,
      },
    }),
    jwt: ({ token, user }) => {
      if (user) {
        token.id = user.id;
      }
      return token;
    },
  },
} satisfies NextAuthConfig;
