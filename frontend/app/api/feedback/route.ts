const BACKEND_URL = (process.env.BACKEND_URL ?? "").replace(/\/$/, "");
const SERVICE_TOKEN = process.env.SERVICE_TOKEN;

type FeedbackPayload = {
  chatId: string;
  messageId: string;
  rating: "positive" | "negative";
  comment?: string;
};

export async function POST(req: Request) {
  const body = (await req.json()) as FeedbackPayload;
  const userId = req.headers.get("x-user-id");
  const userAnonymous = req.headers.get("x-user-anonymous");

  const response = await fetch(`${BACKEND_URL}/feedback`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(SERVICE_TOKEN ? { "X-Service-Token": SERVICE_TOKEN } : {}),
      ...(userId ? { "x-user-id": userId } : {}),
      ...(userAnonymous ? { "x-user-anonymous": userAnonymous } : {}),
    },
    body: JSON.stringify({
      chat_id: body.chatId,
      message_id: body.messageId,
      rating: body.rating,
      comment: body.comment,
    }),
  });

  return new Response(await response.text(), {
    status: response.status,
    headers: {
      "Content-Type": "application/json",
    },
  });
}
