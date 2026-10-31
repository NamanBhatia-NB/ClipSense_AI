export default function PrivacyPage() {
    return (
        <div className="container mx-auto max-w-4xl px-4 py-12">
            <h1 className="text-4xl font-bold mb-8">Privacy Policy</h1>

            <div className="prose prose-slate max-w-none space-y-6">
                <section>
                    <h2 className="text-2xl font-semibold mb-4">1. Information We Collect</h2>
                    <ul className="list-disc pl-6 text-slate-600 space-y-2">
                        <li><strong>Account Information:</strong> Email address, name, and encrypted password.</li>
                        <li><strong>Content:</strong> Videos you upload for processing and the generated clips.</li>
                        <li><strong>Usage Data:</strong> Processing history and clip generation metrics.</li>
                        <li><strong>Payment Information:</strong> Processed securely through Polar.sh (we do not store credit card details).</li>
                    </ul>
                </section>

                <section>
                    <h2 className="text-2xl font-semibold mb-4">2. How We Use Your Information</h2>
                    <ul className="list-disc pl-6 text-slate-600 space-y-2">
                        <li>To provide our AI multimodal highlight generation service.</li>
                        <li>To process payments and manage your credit balance.</li>
                        <li>To communicate service updates and provide customer support.</li>
                        <li>To monitor, maintain, debug, and improve the functionality, reliability, and security of our service.</li>
                        <li>Google user data obtained through Google OAuth is used strictly for authentication and sign-in.</li>
                        <li>We do not use user data to train generalized artificial intelligence or machine learning models.</li>
                        <li>We do not sell user data or use it for advertising, marketing, or profiling.</li>
                    </ul>
                </section>

                <section>
                    <h2 className="text-2xl font-semibold mb-4">3. Data Storage and Security</h2>
                    <ul className="list-disc pl-6 text-slate-600 space-y-2">
                        <li>Uploaded videos and rendered clips are processed locally and securely on the server.</li>
                        <li>All data is encrypted in transit and at rest.</li>
                        <li>Passwords are hashed using industry-standard algorithms.</li>
                    </ul>
                </section>

                <section>
                    <h2 className="text-2xl font-semibold mb-4">5. AI Processing Disclaimer</h2>
                    <p className="text-slate-600">
                        Your uploaded videos are processed by AI models (including third-party services like Google Gemini and WhisperX) to generate transcriptions and clips. By using our service, you consent to this processing.
                    </p>
                </section>

                <section>
                    <h2 className="text-2xl font-semibold mb-4">6. Data Deletion & Revoking Access</h2>
                    <p className="text-slate-600 mb-2">
                        You have full control over your data. If you wish to delete your data or revoke our access to your social media accounts:
                    </p>
                    <ul className="list-disc pl-6 text-slate-600 space-y-2">
                        <li><strong>Disconnecting Accounts:</strong> You can disconnect your Meta or Google accounts at any time from within our app dashboard. This immediately deletes the OAuth tokens from our database.</li>
                        <li><strong>External Revocation:</strong> You can also revoke our app&apos;s access directly from your <a href="https://www.facebook.com/settings?tab=business_tools" target="_blank" rel="noreferrer" className="text-blue-600 hover:underline">Facebook Business Integrations</a> or <a href="https://myaccount.google.com/permissions" target="_blank" rel="noreferrer" className="text-blue-600 hover:underline">Google Security Settings</a>.</li>
                        <li><strong>Account Deletion:</strong> You can request full account deletion, which will permanently wipe your videos, clips, and tokens from our servers.</li>
                    </ul>
                </section>

                <section>
                    <h2 className="text-2xl font-semibold mb-4">7. Third-Party Services</h2>
                    <ul className="list-disc pl-6 text-slate-600 space-y-2">
                        <li><strong>Meta Platforms, Inc:</strong> Social media publishing APIs</li>
                        <li><strong>Google LLC:</strong> YouTube publishing APIs & Gemini AI analysis</li>
                        <li><strong>AWS:</strong> Cloud hosting and secure S3 file storage</li>
                        <li><strong>Polar.sh:</strong> Payment processing</li>
                        <li><strong>Inngest:</strong> Background job scheduling (for delayed publishing)</li>
                        <li><strong>WhisperX:</strong> Audio transcription</li>
                    </ul>
                </section>

                <section>
                    <h2 className="text-2xl font-semibold mb-4">
                        8. Google User Data Sharing and Disclosure
                    </h2>

                    <ul className="list-disc pl-6 text-slate-600 space-y-2">
                        <li>
                            We do not sell, rent, trade, or share Google user data for advertising or marketing purposes.
                        </li>

                        <li>
                            Google user data obtained through Google APIs is used solely to provide the functionality of ClipSense.
                        </li>

                        <li>
                            Google user data may be processed by service providers necessary for operating our service, including:
                            AWS (cloud infrastructure and storage),
                            Google Gemini (AI-powered clip generation workflows),
                            and other infrastructure providers required to deliver the service.
                        </li>

                        <li>
                            These providers may only access data as necessary to perform services on our behalf and are required to protect the data.
                        </li>

                        <li>
                            We may disclose information if required by law, legal process, or to protect the rights, safety, and security of our users and services.
                        </li>
                    </ul>
                </section>

                <section>
                    <h2 className="text-2xl font-semibold mb-4">
                        9. Data Retention
                    </h2>

                    <p className="text-slate-600">
                        We retain user data only for as long as necessary to provide our services,
                        comply with legal obligations, resolve disputes, and enforce agreements.
                    </p>

                    <p className="text-slate-600 mt-2">
                        Users may request deletion of their account and associated data at any time.
                        Upon deletion, Google OAuth tokens, uploaded content, generated clips, and associated metadata
                        will be removed from our systems except where retention is required by law.
                    </p>
                </section>

                <section>
                    <h2 className="text-2xl font-semibold mb-4">10. Cookies</h2>
                    <p className="text-slate-600">
                        We use essential cookies for authentication and session management. No tracking or advertising cookies are used.
                    </p>
                </section>

                <section>
                    <h2 className="text-2xl font-semibold mb-4">11. Contact</h2>
                    <p className="text-slate-600">
                        For privacy concerns, data requests, or to request account deletion, please contact us at: <strong>contact@clipsense.ai</strong>.
                    </p>
                </section>

                <p className="text-sm text-slate-500 mt-8">
                    Last updated: {new Date().toLocaleDateString()}
                </p>
            </div>
        </div>
    );
}