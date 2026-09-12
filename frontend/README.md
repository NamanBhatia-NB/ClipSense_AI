# Connect with Meta (For Invite Users) :

If you are only building this for 2 or 3 specific users right now, keeping it in **Development Mode** as an invite-only tool is actually a very smart, time-saving move. You can skip the Meta App Review entirely.

However, because you are in Development Mode, Meta’s security system is what is causing Account B to return `{"data": []}`. Even if Account B is added as a Tester, Meta’s firewall often blocks the API from seeing the page if the page is locked inside a different Business Portfolio.

Since you only have 2 or 3 users, we don't need to write complex code to fix this. We can use a **manual "VIP Setup"** for these specific users. 

Here is the exact, foolproof checklist to make Account B (and User C) work perfectly in Development Mode.

### Step 1: The "Tester Acceptance" Check
A very common mistake is adding someone as a Tester, but they never actually accept the invite. If the invite is pending, Meta lets them log in, but silently returns `[]` for their pages.
1. Log into Facebook on a computer as **Account B**.
2. Go to [developers.facebook.com](https://developers.facebook.com/) (Account B may need to click "Get Started" to register as a developer).
3. In the top right, click **My Apps**. 
4. Look for a notification or an invite for your **AI Podcast Clipper** app and **Accept** it.

### Step 2: The "Page Admin" Hack (The Ultimate Bypass)
Since this is only for 2 or 3 users, this is the absolute most reliable way to bypass Meta's Business Manager firewall in Development Mode. Instead of fighting the Business settings, just add Account A directly to the Pages.

1. Have **Account B** go to their Facebook Page Settings -> **Page Access**.
2. Under "People with Facebook access", click **Add New**.
3. Add **Account A** (your main developer account) and give it **Full Control**.
4. *(Do the same for any other users you invite. Just have them add your Account A as an Admin to their pages).*

**Why this works:** Because Account A owns the Meta App, any page that Account A is an Admin of is automatically trusted by the App's firewall. 
* Note: Account B will **still** log into your SaaS using their own Account B email. Their clips will still be saved under their own SaaS profile. But when they click "Link Meta", the API will finally see the page because it is co-owned by the developer.

### Step 3: Clear the Cache and Re-Link
Because we are changing permissions natively on Facebook, we need to flush the old, broken token out of your app.

1. Have **Account B** go to their Facebook **Settings -> Business Integrations** and **Remove** the AI Podcast Clipper app.
2. In your local terminal, run `npx prisma studio` and delete the Facebook row for Account B.
3. Log into your SaaS website as Account B.
4. Click **Link Meta**.
5. Check the boxes for the Startup Snippet page!

### Let's verify it without running the Next.js app:
Before you even click publish on a video, let's make sure the API finally sees the page. 

1. Go to the **[Meta Graph API Explorer](https://developers.facebook.com/tools/explorer/)**.
2. Log in as **Account B**.
3. Select your App on the right, and generate a User Token.
4. Type `me/accounts` and hit submit.

If you followed Step 1 (Accepted the Tester Invite) and Step 2 (Added Account A as a Page Admin), that JSON box will finally spit out Account B's page data. 

Let me know the second you see Account B's page in that Graph API explorer!