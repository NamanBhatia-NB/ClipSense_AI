import { NextResponse } from "next/server";
import { Webhook } from "standardwebhooks";
import { db } from "~/server/db";
import { env } from "~/env";

interface PolarMetadata {
    userId?: string;
}

interface PolarOrder {
    id: string;
    order_id?: string;
    checkout_id?: string;
    product_id: string;
    productId?: string;
    metadata?: PolarMetadata;
    user_metadata?: PolarMetadata;
}

interface PolarEvent {
    type: string;
    data: PolarOrder;
}

export async function POST(req: Request) {
    try {
        const bodyText = await req.text();
        if (!bodyText) {
            return new NextResponse("No body provided", { status: 400 });
        }

        const secret = env.POLAR_WEBHOOK_SECRET.replace("polar_whs_", "");
        const webhook = new Webhook(secret);

        const headers = Object.fromEntries(req.headers.entries())as Record<string, string>;
        
        try {
            await webhook.verify(bodyText, headers);
            console.log("✅ Webhook signature verified");
        } catch (error) {
            console.error("❌ Signature Verification Failed:", error);
        }

        const event = JSON.parse(bodyText) as PolarEvent ;
        
        if (event.type === "order.paid") {
            const order = event.data;
            
            const orderId = order.id ?? order.order_id ?? order.checkout_id;
            const userId = order.metadata?.userId ?? order.user_metadata?.userId; 
            const productId = order.product_id ?? order.productId;
            
            if (!userId) {
                console.error("Missing userId in metadata");
                return new NextResponse("Missing userId", { status: 400 });
            }
            
            if (!orderId) {
                console.error("Missing orderId");
                return new NextResponse("Missing orderId", { status: 400 });
            }

            // Check if order already processed
            const existingOrder = await db.processedOrder.findUnique({
                where: { orderId },
            });

            if (existingOrder) {
                console.log(`⚠️ Order ${orderId} already processed. Skipping.`);
                return new NextResponse("Order already processed", { status: 200 });
            }

            console.log(`✅ Order ${orderId} is new, processing...`);

            let creditsToAdd = 0;

            if (productId === env.POLAR_PRODUCT_SMALL_ID) creditsToAdd = 50;
            else if (productId === env.POLAR_PRODUCT_MEDIUM_ID) creditsToAdd = 150;
            else if (productId === env.POLAR_PRODUCT_LARGE_ID) creditsToAdd = 500;

            console.log(`💳 Credits to add: ${creditsToAdd}`);

            if (creditsToAdd > 0) {
                await db.$transaction([
                    db.user.update({
                        where: { id: userId },
                        data: { credits: { increment: creditsToAdd } },
                    }),
                    db.processedOrder.create({
                        data: {
                            orderId,
                            userId,
                            credits: creditsToAdd,
                        },
                    }),
                ]);
                
                console.log(`✅ Added ${creditsToAdd} credits to ${userId} for order ${orderId}`);
            } else {
                console.log(`❌ No credits to add - product ID not matched`);
            }
        }

        return new NextResponse("Webhook processed", { status: 200 });

    } catch (error) {
        console.error("Webhook Error:", error);
        return new NextResponse("Internal Server Error", { status: 500 });
    }
}