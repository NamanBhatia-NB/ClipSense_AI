export default function TermsPage() {
    return (
        <div className="container mx-auto max-w-4xl px-4 py-12">
            <h1 className="text-4xl font-bold mb-8">Terms and Conditions</h1>
            
            <div className="prose prose-slate max-w-none space-y-6">
                <section>
                    <h2 className="text-2xl font-semibold mb-4">1. Acceptance of Terms</h2>
                    <p className="text-slate-600">
                        By accessing and using AI Podcast Clipper, you accept and agree to be bound by the terms and provision of this agreement.
                    </p>
                </section>

                <section>
                    <h2 className="text-2xl font-semibold mb-4">2. AI-Generated Content Disclaimer</h2>
                    <p className="text-slate-600 mb-2">
                        <strong>IMPORTANT:</strong> Our service uses artificial intelligence to automatically generate clips, captions, and transcriptions. Please be aware:
                    </p>
                    <ul className="list-disc pl-6 text-slate-600 space-y-2">
                        <li>AI-generated content may contain errors, inaccuracies, or misinterpretations</li>
                        <li>Transcriptions and captions may not be 100% accurate</li>
                        <li>Clip selection is automated and may not always capture the intended context</li>
                        <li>You are responsible for reviewing and verifying all generated content before use or publishing</li>
                        <li>We are not liable for any damages resulting from inaccurate AI-generated content</li>
                    </ul>
                </section>

                <section>
                    <h2 className="text-2xl font-semibold mb-4">3. Social Media Integrations & Publishing</h2>
                    <p className="text-slate-600 mb-2">
                        Our service allows you to link third-party social media accounts (such as YouTube, Facebook, and Instagram) to publish and schedule content directly. By using these features, you agree that:
                    </p>
                    <ul className="list-disc pl-6 text-slate-600 space-y-2">
                        <li>You grant us permission to post content to your connected accounts on your behalf, at the times you schedule.</li>
                        <li>You remain solely responsible for any content published to your accounts through our service.</li>
                        <li>You must comply with the Terms of Service and Community Guidelines of any connected platform (e.g., Meta, Google/YouTube).</li>
                        <li>We use official APIs (such as YouTube API Services). By using our YouTube integration, you are agreeing to be bound by the <a href="https://www.youtube.com/t/terms" target="_blank" rel="noreferrer" className="text-blue-600 hover:underline">YouTube Terms of Service</a>.</li>
                        <li>We are not responsible if a third-party platform restricts, suspends, or bans your account due to the content you publish or the frequency of your posting.</li>
                    </ul>
                </section>

                <section>
                    <h2 className="text-2xl font-semibold mb-4">4. User Responsibilities</h2>
                    <ul className="list-disc pl-6 text-slate-600 space-y-2">
                        <li>You must own or have explicit rights to all content you upload for processing</li>
                        <li>You are responsible for ensuring uploaded and published content complies with all applicable laws</li>
                        <li>You agree not to upload or publish illegal, harmful, offensive, or copyrighted content</li>
                        <li>You are responsible for keeping your account credentials and connected OAuth tokens secure</li>
                        <li>You must review all AI-generated clips before publishing or sharing</li>
                    </ul>
                </section>

                <section>
                    <h2 className="text-2xl font-semibold mb-4">5. Credits and Payments</h2>
                    <ul className="list-disc pl-6 text-slate-600 space-y-2">
                        <li>Credits are non-refundable once purchased</li>
                        <li>Credits never expire</li>
                        <li>1 credit = 1 generated clip</li>
                        <li>Failed processing may result in credit refunds at our discretion</li>
                        <li>Payments are processed securely through Polar.sh</li>
                    </ul>
                </section>

                <section>
                    <h2 className="text-2xl font-semibold mb-4">6. Service Availability</h2>
                    <p className="text-slate-600">
                        We strive for 99% uptime but do not guarantee uninterrupted service. We are not liable for service interruptions, failed scheduled posts, delayed publishing due to API limits, or data loss.
                    </p>
                </section>

                <section>
                    <h2 className="text-2xl font-semibold mb-4">7. Limitation of Liability</h2>
                    <p className="text-slate-600">
                        AI Podcast Clipper and its operators shall not be liable for any indirect, incidental, special, consequential, or punitive damages resulting from your use of the service, including but not loss of data, loss of revenue, or social media account suspension.
                    </p>
                </section>

                <section>
                    <h2 className="text-2xl font-semibold mb-4">8. Changes to Terms</h2>
                    <p className="text-slate-600">
                        We reserve the right to modify these terms at any time. Continued use of the service constitutes acceptance of modified terms.
                    </p>
                </section>

                <section>
                    <h2 className="text-2xl font-semibold mb-4">9. Contact</h2>
                    <p className="text-slate-600">
                        For questions about these terms, please contact us at: <strong>aipodcastclipper@gmail.com</strong>.
                    </p>
                </section>

                <p className="text-sm text-slate-500 mt-8">
                    Last updated: {new Date().toLocaleDateString()}
                </p>
            </div>
        </div>
    );
}